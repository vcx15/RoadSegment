import os
import numpy as np
import cv2
from functools import reduce
import tqdm

def roughly_equal_bool(img, rgb, diff_limit=150):
    bgr = rgb[::-1]
    h, w, _ = img.shape
    bool_map_lst = []
    for bgr_i in range(3):
        img_i = img[:, :, bgr_i]
        color_i = bgr[bgr_i]
        legal_bool = (np.abs(img_i - color_i) < diff_limit)
        bool_map_lst.append(legal_bool)
    bool_map = reduce(lambda x, y:x & y, bool_map_lst)
    return bool_map

if __name__ == "__main__":
    img_dir = r'Bing_16ZoomLevel'
    # img_dir = r'test'
    min_area = 32

    img_paths = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if '_bi' in file and '_bi' in file.replace('_bi', '', 1) or 'vis' in file or 'color' in file:
                continue
            if not '-b-' in file:
                continue
            if '_' in file:
                continue
            img_paths.append(os.path.join(root, file))

    # img_paths = [r'16zoom\Bing_512\25.943226785322437-51.339111328125-16-USELESS-USELESS-512-USELESS-b-lbl0.png']
    for i, img_path in enumerate(tqdm.tqdm(img_paths)):
#        if i > 200:
#            break
        if True:
            
            img = cv2.imread(img_path)
            bi_img = np.zeros(img.shape[:2], dtype=np.bool)
            # road颜色
            road_color_rgb_lst = [
                (255, 254, 237), # 黄色
                (245, 233, 193),
                (251, 247, 222), # 黄色边缘
                (231, 200, 123),
                (230, 199, 119),
                (240, 220, 163),
                (233, 207, 138), 
                (216, 186, 111), # 黄色箭头
                (216, 186, 111),
                (221, 194, 125),
                (255, 255, 255), # 白色
                (210, 200, 178), # 白色箭头
                (224, 216, 197),
                (218, 210, 192),
                (232, 225, 209),
                (232, 211, 249), # 紫色
                (196, 147, 231),
                (224, 197, 235), # 紫色边缘
                (202, 162, 199),
                (216, 185, 223),
                (201, 162, 198),
                (209, 174, 210),
                (206, 172, 202),
                (209, 174, 211),
                (213, 186, 209),
                (224, 207, 220),
                (220, 199, 216),
                (212, 183, 208),
                (211, 182, 207),
                (219, 196, 214),
                (255, 245, 170), # 橙色
                (239, 208, 148),
                (238, 208, 149),
                (249, 232, 162),
                (247, 227, 159),
                (247, 242, 229), # 肉色
                (255, 254, 236), # 淡黄色
                (235, 208, 137),
                (234, 207, 134),
                (237, 214, 150),
                (254, 244, 170),
                (221, 193, 123), # 淡黄色箭头
                (224, 200, 135),
                (232, 203, 130),
                (216, 207, 188),

                
            ]
            for road_color in road_color_rgb_lst:
                road_map = roughly_equal_bool(img, road_color, diff_limit=10)
                bi_img = bi_img | road_map
            # road颜色strict
            road_color_rgb_lst = [
                (234, 225, 206), # 白色边缘_new
                (227, 215, 190),
                (229, 218, 195),
                (235, 230, 219),
                (248, 246, 243),
                (254, 253, 252),
                (229, 218, 195),
                (248, 246, 240),
                (228, 217, 193),
                (232, 224, 208),
                (232, 224, 207),
                (238, 207, 148), # 黄色_new, 黄色箭头
                (238, 207, 148),
                (255, 244, 169),
                (255, 218, 137),
                (255, 243, 167),
                (255, 238, 161),
                (255, 217, 135),
                (255, 224, 144),
                (254, 214, 137), 
                (238, 234, 226), # 灰色箭头_new
                (227, 221, 207),
            ]
            for road_color in road_color_rgb_lst:
                road_map = roughly_equal_bool(img, road_color, diff_limit=5)
                bi_img = bi_img | road_map
            road_color_rgb_lst = [
                (239, 236, 229), # 白色new
                (238, 232, 225),
                (246, 242, 234)
            ]
            for road_color in road_color_rgb_lst:
                road_map = roughly_equal_bool(img, road_color, diff_limit=1)
                bi_img = bi_img | road_map    


            bi_img = bi_img.astype(np.int) * 255

            # background颜色
            # 
            bi_img_bg = np.zeros(img.shape[:2], dtype=np.bool)
            bg_color_rgb_lst = [

            ]
            for bg_color in bg_color_rgb_lst:
                bg_map = roughly_equal_bool(img, bg_color, diff_limit=3)
                bi_img_bg = bi_img_bg | bg_map
            bi_img_bg = ~bi_img_bg
            bi_img_bg = bi_img_bg.astype(np.int) * 255


            final_map = (bi_img_bg.astype(np.bool) & bi_img.astype(np.bool)).astype(np.uint8)
            final_map_bool = final_map.astype(np.bool)

            # 补充白色道路的灰色边。diff_limit设置为较小值。同时只允许原来白色部分四周被补充
            up_down_left_down = (1, 1, 1, 1)
            legal_mask_up = np.pad(final_map_bool, ((up_down_left_down[0], 0), (0, 0)), 'constant', constant_values=0)[:-up_down_left_down[0], :].astype(np.bool)
            legal_mask_down = np.pad(final_map_bool, ((0, up_down_left_down[1]), (0, 0)), 'constant', constant_values=0)[up_down_left_down[1]:, :].astype(np.bool)
            legal_mask_left = np.pad(final_map_bool, ((0, 0), (up_down_left_down[2], 0)), 'constant', constant_values=0)[:, :-up_down_left_down[2]].astype(np.bool)
            legal_mask_right = np.pad(final_map_bool, ((0, 0), (0, up_down_left_down[3])), 'constant', constant_values=0)[:, up_down_left_down[3]:].astype(np.bool)
            legal_mask = final_map_bool | legal_mask_up | legal_mask_down | legal_mask_left | legal_mask_right
            road_color_rgb_lst_strict = [
                # (227, 227, 228),
                (251, 250, 247)
            ]
            for road_color in road_color_rgb_lst_strict:
                road_map = roughly_equal_bool(img, road_color, diff_limit=5)
                final_map_bool = final_map_bool | road_map 
            final_map_bool = final_map_bool & legal_mask

            # 增加background颜色：分割两条相邻道路
            bi_img_bg = np.zeros(img.shape[:2], dtype=np.bool)
            bg_color_rgb_lst = [

            ]
            for bg_color in bg_color_rgb_lst:
                bg_map = roughly_equal_bool(img, bg_color, diff_limit=1)
                bi_img_bg = bi_img_bg | bg_map
            bi_img_bg = ~bi_img_bg
            bi_img_bg = bi_img_bg.astype(np.int) * 255
            final_map = (bi_img_bg.astype(np.bool) & final_map_bool.astype(np.bool)).astype(np.uint8)
            final_map_bool = final_map.astype(np.bool)

            # 罕见的道路颜色
            do_lst = [

            ]
            do = False
            for do_name in do_lst:
                if do_name in img_path:
                    do = True
            if do:
                road_color_rgb_lst_strict = [

                ]
                for road_color in road_color_rgb_lst_strict:
                    road_map = roughly_equal_bool(img, road_color, diff_limit=1)
                    final_map_bool = final_map_bool | road_map 
                final_map_bool = final_map_bool
            final_map =  final_map_bool.astype(np.int) * 255
        
            
            # 去黑点
            ignore_black_lst = [

            ]
            ignore = False
            for ignore_name in ignore_black_lst:
                if ignore_name in img_path:
                    ignore = True
            if not ignore:
                final_map_mask = (final_map / 255).astype(np.int)
                up = np.pad(final_map_mask, ((1, 0), (0, 0)), 'constant', constant_values=0)[:-1, :]
                down = np.pad(final_map_mask, ((0, 1), (0, 0)), 'constant', constant_values=0)[1:, :]
                left = np.pad(final_map_mask, ((0, 0), (1, 0)), 'constant', constant_values=0)[:, :-1]
                right = np.pad(final_map_mask, ((0, 0), (0, 1)), 'constant', constant_values=0)[:, 1:]
                left_up = np.pad(final_map_mask, ((1, 0), (1, 0)), 'constant', constant_values=0)[:-1, :-1] # 处理左上方块
                left_down = np.pad(final_map_mask, ((0, 1), (1, 0)), 'constant', constant_values=0)[1:, :-1]
                right_up = np.pad(final_map_mask, ((1, 0), (0, 1)), 'constant', constant_values=0)[:-1, 1:]
                right_down = np.pad(final_map_mask, ((0, 1), (0, 1)), 'constant', constant_values=0)[1:, 1:]
                sum_map = up + down + left + right + left_up + left_down + right_down + right_up
                black2white_mask = sum_map > 4
                self_black_map = final_map_mask.astype(np.bool)
                black2white_mask = (black2white_mask & ~self_black_map) | self_black_map
                after_map = black2white_mask.astype(np.int) * 255
                final_map = after_map

                # final_map_mask = (final_map / 255).astype(np.int)
                # up = np.pad(final_map_mask, ((1, 0), (0, 0)), 'constant', constant_values=0)[:-1, :]
                # down = np.pad(final_map_mask, ((0, 1), (0, 0)), 'constant', constant_values=0)[1:, :]
                # left = np.pad(final_map_mask, ((0, 0), (1, 0)), 'constant', constant_values=0)[:, :-1]
                # right = np.pad(final_map_mask, ((0, 0), (0, 1)), 'constant', constant_values=0)[:, 1:]
                # left_up = np.pad(final_map_mask, ((1, 0), (1, 0)), 'constant', constant_values=0)[:-1, :-1] # 处理左上方块
                # left_down = np.pad(final_map_mask, ((0, 1), (1, 0)), 'constant', constant_values=0)[1:, :-1]
                # right_up = np.pad(final_map_mask, ((1, 0), (0, 1)), 'constant', constant_values=0)[:-1, 1:]
                # right_down = np.pad(final_map_mask, ((0, 1), (0, 1)), 'constant', constant_values=0)[1:, 1:]
                # sum_map = up + down + left + right + left_up + left_down + right_down + right_up
                # black2white_mask = sum_map > 5
                # self_black_map = final_map_mask.astype(np.bool)
                # black2white_mask = (black2white_mask & ~self_black_map) | self_black_map
                # after_map = black2white_mask.astype(np.int) * 255
                # final_map = after_map
            # 去白点
                final_map_mask = (final_map / 255).astype(np.int)
                up = np.pad(final_map_mask, ((1, 0), (0, 0)), 'constant', constant_values=0)[:-1, :]
                down = np.pad(final_map_mask, ((0, 1), (0, 0)), 'constant', constant_values=0)[1:, :]
                left = np.pad(final_map_mask, ((0, 0), (1, 0)), 'constant', constant_values=0)[:, :-1]
                right = np.pad(final_map_mask, ((0, 0), (0, 1)), 'constant', constant_values=0)[:, 1:]
                left_up = np.pad(final_map_mask, ((1, 0), (1, 0)), 'constant', constant_values=0)[:-1, :-1] # 处理左上方块
                left_down = np.pad(final_map_mask, ((0, 1), (1, 0)), 'constant', constant_values=0)[1:, :-1]
                right_up = np.pad(final_map_mask, ((1, 0), (0, 1)), 'constant', constant_values=0)[:-1, 1:]
                right_down = np.pad(final_map_mask, ((0, 1), (0, 1)), 'constant', constant_values=0)[1:, 1:]
                sum_map = up + down + left + right + left_up + left_down + right_down + right_up
                black2white_mask = sum_map > 2
                self_black_map = final_map_mask.astype(np.bool)
                black2white_mask = black2white_mask & self_black_map
                after_map = black2white_mask.astype(np.int) * 255
                final_map = after_map

            # 过滤小面积白色
            # final_map = (final_map / 255).astype(np.uint8)
            # contours, hierarchy = cv2.findContours(final_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # for contour in contours:
            #     area = cv2.contourArea(contour)
            #     if area < min_area:
            #         cv2.drawContours(final_map, contour, contourIdx=-1, color=0, thickness=-1)
            # final_map = final_map.astype(np.int) * 255

            # 过滤小面积黑色
            # final_map_temp = (~(final_map / 255).astype(np.bool)).astype(np.uint8)
            # contours, hierarchy = cv2.findContours(final_map_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # for contour in contours:
            #     area = cv2.contourArea(contour)
            #     if area < min_area:
            #         cv2.drawContours(final_map_temp, contour, contourIdx=-1, color=0, thickness=-1)
            # final_map = ~final_map_temp.astype(np.bool)
            # final_map = final_map.astype(np.int) * 255

            # 去黑点
            final_map_mask = (final_map / 255).astype(np.int)
            up = np.pad(final_map_mask, ((1, 0), (0, 0)), 'constant', constant_values=0)[:-1, :]
            down = np.pad(final_map_mask, ((0, 1), (0, 0)), 'constant', constant_values=0)[1:, :]
            left = np.pad(final_map_mask, ((0, 0), (1, 0)), 'constant', constant_values=0)[:, :-1]
            right = np.pad(final_map_mask, ((0, 0), (0, 1)), 'constant', constant_values=0)[:, 1:]
            left_up = np.pad(final_map_mask, ((1, 0), (1, 0)), 'constant', constant_values=0)[:-1, :-1] # 处理左上方块
            left_down = np.pad(final_map_mask, ((0, 1), (1, 0)), 'constant', constant_values=0)[1:, :-1]
            right_up = np.pad(final_map_mask, ((1, 0), (0, 1)), 'constant', constant_values=0)[:-1, 1:]
            right_down = np.pad(final_map_mask, ((0, 1), (0, 1)), 'constant', constant_values=0)[1:, 1:]
            sum_map = up + down + left + right + left_up + left_down + right_down + right_up
            black2white_mask = sum_map > 4
            self_black_map = final_map_mask.astype(np.bool)
            black2white_mask = (black2white_mask & ~self_black_map) | self_black_map
            after_map = black2white_mask.astype(np.int) * 255
            final_map = after_map  

            final_map_mask = (final_map / 255).astype(np.int)
            up = np.pad(final_map_mask, ((1, 0), (0, 0)), 'constant', constant_values=0)[:-1, :]
            down = np.pad(final_map_mask, ((0, 1), (0, 0)), 'constant', constant_values=0)[1:, :]
            left = np.pad(final_map_mask, ((0, 0), (1, 0)), 'constant', constant_values=0)[:, :-1]
            right = np.pad(final_map_mask, ((0, 0), (0, 1)), 'constant', constant_values=0)[:, 1:]
            left_up = np.pad(final_map_mask, ((1, 0), (1, 0)), 'constant', constant_values=0)[:-1, :-1] # 处理左上方块
            left_down = np.pad(final_map_mask, ((0, 1), (1, 0)), 'constant', constant_values=0)[1:, :-1]
            right_up = np.pad(final_map_mask, ((1, 0), (0, 1)), 'constant', constant_values=0)[:-1, 1:]
            right_down = np.pad(final_map_mask, ((0, 1), (0, 1)), 'constant', constant_values=0)[1:, 1:]
            sum_map = up + down + left + right + left_up + left_down + right_down + right_up
            black2white_mask = sum_map > 5
            self_black_map = final_map_mask.astype(np.bool)
            black2white_mask = (black2white_mask & ~self_black_map) | self_black_map
            after_map = black2white_mask.astype(np.int) * 255
            final_map = after_map  

            img_ = img.copy()
            img[final_map == 255] = (0, 0, 255)
            road_mask = (final_map == 255)
            color_img = (road_mask[:, :, None] * img_).astype(np.uint8)
        #    cv2.imwrite(img_path.replace('.png', '_color_vis.png'), color_img)
            cv2.imwrite(img_path.replace('.png', '_bi.png'), final_map)
        #    cv2.imwrite(img_path.replace('.png', '_vis.png'), img)    
       