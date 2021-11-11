import cv2
import pandas as pd
from skimage import morphology
import numpy as np
import glob
from pandas import DataFrame
import os
import copy
import tqdm
import math
import random
from config import map_color_width_info

def vis_road_pts(pts, skeleton, name='temp'):
    img = copy.copy(skeleton)
    if img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pt in pts:
        img[pt[0], pt[1]] = (0, 0, 255)
    print('wirte to {}.png'.format(name))
    cv2.imwrite(name + '.png', img)

def get_road_length(pts, lat, zoom, resolution):
    # pts:[(y, x)...]
    up_pts = pts + [-1, 0]
    down_pts = pts + [1, 0]
    left_pts = pts + [0, -1]
    right_pts = pts + [0, 1]
    temp = up_pts[0]
    total_dis_1 = 0
    for trans_pts in [up_pts, down_pts, left_pts, right_pts]:
        pts_temp = pts[None, :, :]
        trans_pts = trans_pts[:, None, :]
        dis = np.abs(pts_temp - trans_pts)
        dis = np.sum(dis, axis=-1)
        num = (dis == 0).sum()
        total_dis_1 += num
    up_pts = pts + [-1, -1]
    down_pts = pts + [1, 1]
    left_pts = pts + [1, -1]
    right_pts = pts + [-1, 1]
    total_dis_sqrt2 = 0
    for trans_pts in [up_pts, down_pts, left_pts, right_pts]:
        pts_temp = pts[None, :, :]
        trans_pts = trans_pts[:, None, :]
        dis = np.abs(pts_temp - trans_pts)
        dis = np.sum(dis, axis=-1)
        num = (dis == 0).sum()
        total_dis_sqrt2 += num
    total_dis = (total_dis_1 + total_dis_sqrt2 * 2**0.5) / 2 + 1 # +1:num_ignore
    metersPerPx = 156543.03392 * math.cos(lat * math.pi / 180) / math.pow(2, zoom)    
    real_dis_meter = total_dis * metersPerPx / (resolution / 256)
    return real_dis_meter, metersPerPx / (resolution / 256)



def generate_excel(header, data, xlsx_path):
    df = DataFrame(data, columns=header)
    df.to_excel(xlsx_path, index=False)

def sum_around(skeleton_pad):
    up = np.pad(skeleton_pad, ((1, 0), (0, 0)), 'constant', constant_values=0)[:-1, :]
    down = np.pad(skeleton_pad, ((0, 1), (0, 0)), 'constant', constant_values=0)[1:, :]
    left = np.pad(skeleton_pad, ((0, 0), (1, 0)), 'constant', constant_values=0)[:, :-1]
    right = np.pad(skeleton_pad, ((0, 0), (0, 1)), 'constant', constant_values=0)[:, 1:]
    left_up = np.pad(skeleton_pad, ((1, 0), (1, 0)), 'constant', constant_values=0)[:-1, :-1] # 处理左上方块
    left_down = np.pad(skeleton_pad, ((0, 1), (1, 0)), 'constant', constant_values=0)[1:, :-1]
    right_up = np.pad(skeleton_pad, ((1, 0), (0, 1)), 'constant', constant_values=0)[:-1, 1:]
    right_down = np.pad(skeleton_pad, ((0, 1), (0, 1)), 'constant', constant_values=0)[1:, 1:]
    sum_map = up + down + left + right + left_up + left_down + right_down + right_up + skeleton_pad
    return sum_map

def skeleton_gen(img):
    img[img == 255] = 1
    skeleton0 = morphology.skeletonize(img)
    skeleton = skeleton0.astype(np.uint8)*255
    return skeleton

def get_road_intersection(skeleton):
    skeleton_pad = skeleton.astype(np.int)
    sum_map = sum_around(skeleton_pad)
    node_map = (sum_map >= 4 * 255) & skeleton.astype(np.bool)
    node_map_grey = node_map.astype(np.uint8) * 255
    node_idx = np.where(node_map)
    num_nodes, labels, stats, centroids = cv2.connectedComponentsWithStats(node_map_grey, connectivity=8)
    node_center_cor_lst = []
    for node_i in range(1, num_nodes):
        node_idx = np.where(labels == node_i)
        node_pts = np.array(node_idx).T # ((y1, x1), (y2, x2)...)
        dis_lst = []
        for pt_i in range(len(node_pts)):
            pt = node_pts[pt_i]
            distance = pt - node_pts
            dis_sum = np.abs(distance).sum()
            dis_lst.append(dis_sum)
        center_i = np.array(dis_lst).argmin()
        node_center_cor_lst.append(node_pts[center_i])

    # 小区域内有多个节点，随机只选择一个
    dis_threshod = 10
    new_center_lst = []
    node_i_flag_lst = []
    for node_i in range(len(node_center_cor_lst)):
        if node_i in node_i_flag_lst:
            continue
        temp = [node_i]
        for node_j in range(node_i + 1, len(node_center_cor_lst)):
            if node_j in node_i_flag_lst:
                continue
            dis = np.linalg.norm(node_center_cor_lst[node_i] - node_center_cor_lst[node_j], ord=2)
            if dis < dis_threshod:
                node_i_flag_lst.append(node_j)
                temp.append(node_j)
        new_center_lst.append(node_center_cor_lst[random.choice(temp)])
    node_center_cor_lst = new_center_lst


    
    node_center_bgr_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    for node_center_cor in node_center_cor_lst: 
        node_center_bgr_skeleton[node_center_cor[0], node_center_cor[1]] = (0, 0, 255)
        cv2.circle(node_center_bgr_skeleton, tuple(reversed(node_center_cor)), 5, (0, 255, 255), 2) # cv2.circle: (x, y)
    return node_center_bgr_skeleton, node_center_cor_lst # (y, x)

def get_subroad_info(skeleton, skeleton_node_cor_lst, img, gray, mask_path, resolution=256, raw_lat_long=None):
    lat = raw_lat_long[0]
    zoom = 14 if '-14-' in mask_path else 16
    if skeleton_node_cor_lst != []:
        node_cor = np.array(skeleton_node_cor_lst)
        # 交点附近grey赋1，skeleton赋0
        num_ignore = 2
        node_cor_lst = [node_cor + [i, j] for i in range(-num_ignore, num_ignore) for j in range(-num_ignore, num_ignore)]
        node_cor_heavy_lst = [node_cor + [i, j] for i in range(-num_ignore, num_ignore) for j in range(-num_ignore, num_ignore)]
        node_cor_lst = [np.clip(a, 0, resolution - 1) for a in node_cor_lst]
        node_cor_heavy_lst = [np.clip(a, 0, resolution - 1) for a in node_cor_heavy_lst]
        for node_cor in node_cor_lst:
            for pt in node_cor:
                skeleton[pt[0], pt[1]] = 0
        for node_cor in node_cor_heavy_lst:
            for pt in node_cor:
                gray[pt[0], pt[1]] = 1
    else:
        num_ignore = 0                
    # cv2.imwrite('skeleton.png', skeleton)
    # cv2.imwrite('gray.png', gray * 255)
    # 连通域排序
    num_nodes, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    subroad_pts_lst = []
    for node_i in range(1, num_nodes):
        node_idx = np.where(labels == node_i)
        node_pts = np.array(node_idx).T # ((y1, x1), (y2, x2)...)
        mean_pt = node_pts.mean(axis=0) # (y, x)
        subroad_pts_lst.append([node_pts, mean_pt[0] * 1000 + mean_pt[1]])
    sort_subroad_pts_lst = sorted(subroad_pts_lst, key=lambda x:x[1])
    sort_subroad_pts_lst = [t[0] for t in sort_subroad_pts_lst]
    # subroad长度
    subroad_len_lst = []
    for subroad_i in range(len(sort_subroad_pts_lst)):
        subroad_pts = sort_subroad_pts_lst[subroad_i]
        subroad_len, meterPerpix = get_road_length(subroad_pts, lat, zoom, resolution)
        subroad_len_lst.append(subroad_len)
    # subroad宽度
    bg_pts = np.where(gray == 0)
    bg_pts = np.array(bg_pts).T # ((y1, x1), (y2, x2)...)
    bg_pts = bg_pts[None, ...]
    subroad_width_lst = []
    for subroad_pts in sort_subroad_pts_lst:
        subroad_pts = subroad_pts[:, None, :]
        # 根据subroad_pts的坐标，过滤离得远的bg_pts
        min_yx = np.clip(subroad_pts.min(axis=(0, 1)) - 40, 0, resolution)
        max_yx = np.clip(subroad_pts.max(axis=(0, 1)) + 40, 0, resolution)
        legal_bg_idx = (bg_pts < max_yx) & (bg_pts > min_yx)
        legal_bg_idx = legal_bg_idx[0, :, 0] & legal_bg_idx[0, :, 1]
        legal_bg = bg_pts[:, legal_bg_idx, :]
        if legal_bg.shape[1] <= 256 * 256:
            dis_matrix = np.linalg.norm(legal_bg - subroad_pts, ord=2, axis=2)
            min_dis = np.min(dis_matrix, axis=1)
        else:
            min_dis = np.zeros(subroad_pts.shape[0])
            split_num = 16
            subroad_pts_lst_unit = subroad_pts.shape[0] // split_num
            subroad_pts_lst = [subroad_pts[i*subroad_pts_lst_unit:(i+1)*subroad_pts_lst_unit, 0, :] for i in range(split_num)]            
            for i, subroad_pts_part in enumerate(subroad_pts_lst):
                dis_matrix_part = np.linalg.norm(legal_bg - subroad_pts_part[:, None, :], ord=2, axis=2)
                min_dis_part = np.min(dis_matrix_part, axis=1)
                min_dis[i*subroad_pts_lst_unit:(i+1)*subroad_pts_lst_unit] = min_dis_part
        # start, end = 3/10, 7/10
        # dis_lst = np.sort(min_dis)
        # if len(dis_lst) > 1:
        #     dis_lst = np.sort(min_dis)[int(start * len(min_dis)):int(end * len(min_dis))]
        # subroad_width_lst.append(np.mean(dis_lst) * 2)
        dis_lst = np.sort(min_dis)
        width = 2 * dis_lst[len(dis_lst) // 2]
        subroad_width_lst.append(width)

    # 过滤过短的道路
    subroad_min_length = 2 * resolution / 256
    subraod_legal_lst = []
    for subroad_i, length in enumerate(subroad_len_lst):
        if length > subroad_min_length:
            subraod_legal_lst.append(True)
        else:
            subraod_legal_lst.append(False)
    new_subroad_width_lst, new_subroad_len_lst, new_sort_subroad_pts_lst = [], [], []
    for subroad_i, legal_flag in enumerate(subraod_legal_lst):
        if legal_flag:
            new_subroad_len_lst.append(subroad_len_lst[subroad_i])
            new_subroad_width_lst.append(subroad_width_lst[subroad_i])
            new_sort_subroad_pts_lst.append(sort_subroad_pts_lst[subroad_i])
    subroad_len_lst = new_subroad_len_lst
    subroad_width_lst = new_subroad_width_lst
    sort_subroad_pts_lst = new_sort_subroad_pts_lst

    # 过滤过窄的道路
    subroad_min_width = 1 # edit 1
    subraod_legal_lst = []
    for subroad_i, width in enumerate(subroad_width_lst):
        if width > subroad_min_width:
            subraod_legal_lst.append(True)
        else:
            subraod_legal_lst.append(False)
    new_subroad_width_lst, new_subroad_len_lst, new_sort_subroad_pts_lst = [], [], []
    for subroad_i, legal_flag in enumerate(subraod_legal_lst):
        if legal_flag:
            new_subroad_len_lst.append(subroad_len_lst[subroad_i])
            new_subroad_width_lst.append(subroad_width_lst[subroad_i])
            new_sort_subroad_pts_lst.append(sort_subroad_pts_lst[subroad_i])
    subroad_len_lst = new_subroad_len_lst
    subroad_width_lst = new_subroad_width_lst
    sort_subroad_pts_lst = new_sort_subroad_pts_lst

    # 过滤四周的道路
    ignore_lenth = 10 # edit 2
    subraod_legal_lst = []
    for subroad_i, subroad_pts in enumerate(sort_subroad_pts_lst):
        if (subroad_pts.mean(axis=0) > ignore_lenth).all() and \
            (subroad_pts.mean(axis=0) < resolution - ignore_lenth).all():
            subraod_legal_lst.append(True)
        else:
            subraod_legal_lst.append(False)
    new_subroad_width_lst, new_subroad_len_lst, new_sort_subroad_pts_lst = [], [], []
    for subroad_i, legal_flag in enumerate(subraod_legal_lst):
        if legal_flag:
            new_subroad_len_lst.append(subroad_len_lst[subroad_i])
            new_subroad_width_lst.append(subroad_width_lst[subroad_i])
            new_sort_subroad_pts_lst.append(sort_subroad_pts_lst[subroad_i])
    subroad_len_lst = new_subroad_len_lst
    subroad_width_lst = new_subroad_width_lst
    sort_subroad_pts_lst = new_sort_subroad_pts_lst


    # 根据颜色与宽度判断道路类别
    map_type = 'google' if '-g-' in mask_path else 'osm' if '-o-' in mask_path else 'bing'
    map_provide_info = map_color_width_info[map_type]
    subroad_name_lst = []
    if map_type == 'google':
        for subroad_i, subroad_pts in enumerate(sort_subroad_pts_lst):
            cor_y = subroad_pts[:, 0]
            cor_x = subroad_pts[:, 1]
            subimg_bgr = np.mean(img[cor_y, cor_x], axis=0)
            # 获得颜色
            sample_pts = random.sample(subroad_pts.tolist(), min(20, len(subroad_pts)))
            # 扩充颜色选择点
            up_down_left_right_range = 2
            all_sample_pts = []
            for r in range(-up_down_left_right_range, up_down_left_right_range + 1):
                up_lst = [[pt[0] - r, pt[1]] for pt in sample_pts]
                down_lst = [[pt[0] + r, pt[1]] for pt in sample_pts]
                left_lst = [[pt[0], pt[1] - r] for pt in sample_pts]
                right_lst = [[pt[0], pt[1] + r] for pt in sample_pts]
                all_sample_pts.extend(up_lst), all_sample_pts.extend(down_lst), all_sample_pts.extend(left_lst), all_sample_pts.extend(right_lst), 
            sample_pts = np.clip(all_sample_pts, 0, resolution - 1).tolist()            
            subimg_color_name, sample_pts_color_lst = 'white', []
            for sample_pt in sample_pts:
                min_color_dis, pt_color_name, sample_bgr = 1e9, 'white', img[sample_pt[0], sample_pt[1]]
                for color_name, rgb in map_provide_info['color_name'].items():
                    bgr = rgb[::-1]
                    color_dis = np.linalg.norm(sample_bgr - bgr)
                    if color_dis < min_color_dis:
                        pt_color_name = color_name
                        min_color_dis = color_dis
                sample_pts_color_lst.append(pt_color_name)
            subimg_color_name = max(sample_pts_color_lst, key=sample_pts_color_lst.count)
            # 颜色+宽度分类
            min_width_dis, min_width_idx = 1e9, 0
            for type_i, road_typename in enumerate(map_provide_info['name']):
                if subimg_color_name not in road_typename:
                    continue
                width_dis = abs(map_provide_info['width'][type_i] - subroad_width_lst[subroad_i])
                if width_dis < min_width_dis:
                    min_width_dis = width_dis
                    min_width_idx = type_i
            subroad_name = map_provide_info['name'][min_width_idx]
            subroad_name_lst.append(subroad_name)
    elif map_type == 'bing':
        for subroad_i, subroad_pts in enumerate(sort_subroad_pts_lst):
            if subroad_i in (2, 6, 7, 8,9, 10, 11, 13, 17,19,20,25,27):
                debug = 233
                vis_road_pts(subroad_pts, img, name=str(subroad_i))
            cor_y = subroad_pts[:, 0]
            cor_x = subroad_pts[:, 1]
            subimg_bgr = np.mean(img[cor_y, cor_x], axis=0)
            # 获得颜色
            sample_pts = random.sample(subroad_pts.tolist(), min(20, len(subroad_pts)))
            # 扩充颜色选择点
            up_down_left_right_range = 2
            all_sample_pts = []
            for r in range(-up_down_left_right_range, up_down_left_right_range + 1):
                up_lst = [[pt[0] - r, pt[1]] for pt in sample_pts]
                down_lst = [[pt[0] + r, pt[1]] for pt in sample_pts]
                left_lst = [[pt[0], pt[1] - r] for pt in sample_pts]
                right_lst = [[pt[0], pt[1] + r] for pt in sample_pts]
                all_sample_pts.extend(up_lst), all_sample_pts.extend(down_lst), all_sample_pts.extend(left_lst), all_sample_pts.extend(right_lst), 
            sample_pts = np.clip(all_sample_pts, 0, resolution - 1).tolist()
            subimg_color_name, sample_pts_color_lst = 'white', []
            for sample_pt in sample_pts:
                min_color_dis, pt_color_name, sample_bgr = 1e9, 'white', img[sample_pt[0], sample_pt[1]]
                for color_name, rgb in map_provide_info['color_name'].items():
                    bgr = rgb[::-1]
                    color_dis = np.linalg.norm(sample_bgr - bgr)
                    if color_dis < min_color_dis:
                        pt_color_name = color_name
                        min_color_dis = color_dis
                sample_pts_color_lst.append(pt_color_name)
            subimg_color_name = max(sample_pts_color_lst, key=sample_pts_color_lst.count)
            # 颜色+宽度分类
            min_width_dis, min_width_idx = 1e9, 0
            for type_i, road_typename in enumerate(map_provide_info['name']):
                if subimg_color_name not in road_typename:
                    continue
                width_dis = abs(map_provide_info['width'][type_i] - subroad_width_lst[subroad_i])
                if width_dis < min_width_dis:
                    min_width_dis = width_dis
                    min_width_idx = type_i
            subroad_name = map_provide_info['name'][min_width_idx]
            subroad_name_lst.append(subroad_name)
    elif map_type == 'osm':
        for subroad_i, subroad_pts in enumerate(sort_subroad_pts_lst):
            cor_y = subroad_pts[:, 0]
            cor_x = subroad_pts[:, 1]
            subimg_bgr = np.mean(img[cor_y, cor_x], axis=0)
            # 获得颜色
            sample_pts = random.sample(subroad_pts.tolist(), min(20, len(subroad_pts)))
            # 扩充颜色选择点
            up_down_left_right_range = 2
            all_sample_pts = []
            for r in range(-up_down_left_right_range, up_down_left_right_range + 1):
                up_lst = [[pt[0] - r, pt[1]] for pt in sample_pts]
                down_lst = [[pt[0] + r, pt[1]] for pt in sample_pts]
                left_lst = [[pt[0], pt[1] - r] for pt in sample_pts]
                right_lst = [[pt[0], pt[1] + r] for pt in sample_pts]
                all_sample_pts.extend(up_lst), all_sample_pts.extend(down_lst), all_sample_pts.extend(left_lst), all_sample_pts.extend(right_lst), 
            sample_pts = np.clip(all_sample_pts, 0, resolution - 1).tolist()
            subimg_color_name, sample_pts_color_lst = 'white', []
            for sample_pt in sample_pts:
                min_color_dis, pt_color_name, sample_bgr = 1e9, 'white', img[sample_pt[0], sample_pt[1]]
                for color_name, rgb in map_provide_info['color_name'].items():
                    bgr = rgb[::-1]
                    color_dis = np.linalg.norm(sample_bgr - bgr)
                    if color_dis < min_color_dis:
                        pt_color_name = color_name
                        min_color_dis = color_dis
                sample_pts_color_lst.append(pt_color_name)
            subimg_color_name = max(sample_pts_color_lst, key=sample_pts_color_lst.count)
            # 颜色+宽度分类
            min_width_dis, min_width_idx = 1e9, 0
            for type_i, road_typename in enumerate(map_provide_info['name']):
                if subimg_color_name not in road_typename:
                    continue
                width_dis = abs(map_provide_info['width'][type_i] - subroad_width_lst[subroad_i])
                if width_dis < min_width_dis:
                    min_width_dis = width_dis
                    min_width_idx = type_i
            subroad_name = map_provide_info['name'][min_width_idx]
            subroad_name_lst.append(subroad_name)              

    avail_road_type = np.unique(np.array(subroad_name_lst)).tolist()
    avail_road_length = [0 for _ in range(len(avail_road_type))]
    for road_i in range(len(sort_subroad_pts_lst)):
        road_type = subroad_name_lst[road_i]
        road_type_idx = avail_road_type.index(road_type)
        road_len = subroad_len_lst[road_i]
        avail_road_length[road_type_idx] = avail_road_length[road_type_idx] + road_len
    # subroad_info = {'road_type':avail_road_type, 'road_length':avail_road_length, 'total_road_length':[sum(avail_road_length)] + ['' for _ in range(len(avail_road_length) - 1)]}
    # if len(subroad_info['road_length']) == 0:
    #     subroad_info['total_road_length'] = ''
    # if len(subroad_info['road_length']) != 0:
    #     subroad_info['road_name'] = [map_provide_info['name_map'][t] for t in subroad_info['road_type']]
    # else:
    #     subroad_info['road_name'] = ''
    # return subroad_info
    subroad_info = {'road_type':avail_road_type, 'road_length':'', 'road_length_meter':avail_road_length, 'total_road_length_meter':[sum(avail_road_length)] + ['' for _ in range(len(avail_road_length) - 1)], 'total_road_length':''}
    if len(subroad_info['road_length_meter']) == 0:
        subroad_info['total_road_length_meter'] = ''
    else:
        subroad_info['total_road_length'] = ['' for _ in range(len(subroad_info['road_length_meter']))]
        subroad_info['total_road_length'][0] = subroad_info['total_road_length_meter'][0] / meterPerpix
        subroad_info['road_length'] = [t/meterPerpix for t in subroad_info['road_length_meter']]
    if len(subroad_info['road_length']) != 0:
        subroad_info['road_name'] = [map_provide_info['name_map'][t] for t in subroad_info['road_type']]
    else:
        subroad_info['road_name'] = ''
    # edit 3, 如果该颜色较少，那么就删去这个颜色，取成最多的道路类型
    if len(subroad_info['road_length']) != 0:
        min_ratio = 0.01
        delete_index_lst = []
        max_ratio_index = subroad_info['road_length'].index(max(subroad_info['road_length']))
        total_len = subroad_info['total_road_length'][0]
        for subroad_i, subroad_len in enumerate(subroad_info['road_length']):
            if subroad_len / total_len < min_ratio:
                delete_index_lst.append(subroad_i)
        for subroad_i in delete_index_lst:
            subroad_info['road_length'][max_ratio_index] =  \
                subroad_info['road_length'][max_ratio_index] + subroad_info['road_length'][subroad_i]
            subroad_info['road_length_meter'][max_ratio_index] =  \
                subroad_info['road_length_meter'][max_ratio_index] + subroad_info['road_length_meter'][subroad_i]
            subroad_info['road_length'][subroad_i], subroad_info['road_length_meter'][subroad_i] = -1, -1
            subroad_info['road_name'][subroad_i], subroad_info['road_type'][subroad_i] = -1, -1
        subroad_info['road_length'] = list(filter(lambda x:x!=-1, subroad_info['road_length']))
        subroad_info['road_length_meter'] = list(filter(lambda x:x!=-1, subroad_info['road_length_meter']))
        subroad_info['road_name'] = list(filter(lambda x:x!=-1, subroad_info['road_name']))
        subroad_info['road_type'] = list(filter(lambda x:x!=-1, subroad_info['road_type']))
        subroad_info['total_road_length_meter'] = subroad_info['total_road_length_meter'][:len(subroad_info['road_name'])]
        subroad_info['total_road_length'] = subroad_info['total_road_length'][:len(subroad_info['road_name'])]
    return subroad_info    
    
def get_lat_long(skeleton_node_cor_lst, lat_long, resolution=256):
    if skeleton_node_cor_lst:
        lat, long = lat_long[0], [1] # lat:经度, long:纬度
        skeleton_node_cor_lst = np.array(skeleton_node_cor_lst) # (y, x)
        skeleton_node_cor_lst[:, 0] = 256 - skeleton_node_cor_lst[:, 0]
        dis = (skeleton_node_cor_lst - [127.5, 127.5]) * (256 / resolution)
        long_lat_unit = np.array([8.993216192195822e-6, 1.141255544679108e-5])
        long_lat_delta = long_lat_unit * dis
        lat_long_lst = np.array(lat_long) + long_lat_delta[:, ::-1]
        return lat_long_lst.tolist()
    else: 
        return ''

def get_resolution(img_path):
    path = os.path.basename(img_path)
    resolution = 256
    if '-256-' in path:
        resolution = 256
    elif '-512-' in path:
        resolution = 512
    elif '-1024-' in path:
        resolution = 1024
    return resolution

def get_raw_lat_long(mask_path):
    mask_name = os.path.basename(mask_path)
    idx_lst = []
    for c_i, c in enumerate(mask_name):
        if c == '.':
            idx_lst.append(c_i)
    lat_str = mask_name[:idx_lst[0]] + '.'
    for c in mask_name[idx_lst[0] + 1:]:
        if c.isdigit():
            lat_str = lat_str + c
        else:
            break
    no_lat_path = mask_name[len(lat_str) + 1:]
    dot_index = no_lat_path.index('.')
    long_str = no_lat_path[:dot_index] + '.'
    for c in no_lat_path[dot_index + 1:]:
        if c.isdigit():
            long_str = long_str + c
        else:
            break
    lat_long = [eval(lat_str), eval(long_str)]
    return lat_long
    
    

if __name__ == "__main__":
    img_dir = r'16zoom\Bing_512' # 第一次运行这个
    # img_dir = r'16zoom\Google_1024' # 第二次运行这个
    # img_dir = r'16zoom\OSM_512' # 第三次运行这个
    img_dir = r'temp'
    img_paths = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if 'bing' not in file:
                if 'vis' in file or '_bi' not in file:
                    continue
            else:
                if 'vis' in file or 'bi.png' not in file:
                    continue                
            img_paths.append(os.path.join(root, file))
    
    # img_paths = [r'16zoom\Bing_512\13.22925095887976-45.3021240234375-16-USELESS-USELESS-512-USELESS-b-lbl0_bi.png']
    for i, mask_path in tqdm.tqdm(enumerate(img_paths)):
        if i > 200:
            break
        raw_lat_long = get_raw_lat_long(mask_path)
        img_path = mask_path.replace('_bi.png', '.png')
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        skeleton = skeleton_gen(gray)
        skeleton_node, skeleton_node_cor_lst = get_road_intersection(skeleton)
        skeleton_node_cor_lst = [list(t) for t in skeleton_node_cor_lst]
        skeleton_node_lat_long_lst = get_lat_long(skeleton_node_cor_lst, raw_lat_long, resolution=get_resolution(img_path))
        _data = {'total_junction_num':[len(skeleton_node_cor_lst)] + ['' for _ in range(len(skeleton_node_cor_lst) - 1)], 'junction_coordinate':skeleton_node_cor_lst, 'lat_long':skeleton_node_lat_long_lst}
        if len(skeleton_node_cor_lst) == 0:
            _data['total_junction_num'] = ''
            _data['lat_long'] = ''
        generate_excel(
            header=['total_junction_num', 'junction_coordinate', 'lat_long'], 
            data=_data,
            xlsx_path=mask_path.replace('_bi.png', '_junction.xlsx')
        )
        subroad_info = get_subroad_info(skeleton, skeleton_node_cor_lst, img, gray, mask_path, resolution=get_resolution(img_path), raw_lat_long=raw_lat_long)

        generate_excel(
            header=['road_type', 'road_name', 'road_length', 'total_road_length', 'road_length_meter', 'total_road_length_meter'], 
            data=subroad_info,
            xlsx_path=mask_path.replace('_bi.png', '_road_info.xlsx')
        )        
        cv2.imwrite(mask_path.replace('bi.png', 'bi_vis.png'), skeleton_node)

        

    