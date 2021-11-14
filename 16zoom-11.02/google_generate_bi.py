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
    img_dir = r'16zoom/Google_1024'
    # img_dir = r'temp'
    min_area = -1

    img_paths = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if '_bi' in file or 'vis' in file or 'color' in file:
                continue
            if not '-g-' in file:
                continue
            if '_' in file:
                continue
            img_paths.append(os.path.join(root, file))

    # img_paths = [r'16zoom\Google_1024\25.110471486223332-55.1788330078125-16-USELESS-USELESS-1024-USELESS-g-lbl0.png']
    for img_i, img_path in enumerate(tqdm.tqdm(img_paths)):
        # if img_i > 50:
        #     break
        if True:
            
            img = cv2.imread(img_path)
            bi_img = np.zeros(img.shape[:2], dtype=np.bool)
            # road颜色
            road_color_rgb_lst = [
                (253, 226, 147), # 橙色
                (249, 171, 0), # 橙色边缘
                (255, 255, 255), # 白色
            ]
            for road_color in road_color_rgb_lst:
                road_map = roughly_equal_bool(img, road_color, diff_limit=40)
                bi_img = bi_img | road_map

            road_color_rgb_lst = [
                (213, 216, 219), # 铁路
            ]
            for road_color in road_color_rgb_lst:
                road_map = roughly_equal_bool(img, road_color, diff_limit=10)
                bi_img = bi_img | road_map                


            bi_img = bi_img.astype(np.int) * 255

            # background颜色
            # 
            bi_img_bg = np.zeros(img.shape[:2], dtype=np.bool)
            bg_color_rgb_lst = [

            ]
            for bg_color in bg_color_rgb_lst:
                bg_map = roughly_equal_bool(img, bg_color, diff_limit=5)
                bi_img_bg = bi_img_bg | bg_map
            bi_img_bg = ~bi_img_bg
            bi_img_bg = bi_img_bg.astype(np.int) * 255


            final_map = (bi_img_bg.astype(np.bool) & bi_img.astype(np.bool)).astype(np.uint8)
            final_map_bool = final_map.astype(np.bool)

            # 补充白色道路的灰色边。diff_limit设置为较小值。同时只允许原来白色部分四周被补充
            up_down_left_down = (2, 2, 2, 2)
            legal_mask_up = np.pad(final_map_bool, ((up_down_left_down[0], 0), (0, 0)), 'constant', constant_values=0)[:-up_down_left_down[0], :].astype(np.bool)
            legal_mask_down = np.pad(final_map_bool, ((0, up_down_left_down[1]), (0, 0)), 'constant', constant_values=0)[up_down_left_down[1]:, :].astype(np.bool)
            legal_mask_left = np.pad(final_map_bool, ((0, 0), (up_down_left_down[2], 0)), 'constant', constant_values=0)[:, :-up_down_left_down[2]].astype(np.bool)
            legal_mask_right = np.pad(final_map_bool, ((0, 0), (0, up_down_left_down[3])), 'constant', constant_values=0)[:, up_down_left_down[3]:].astype(np.bool)
            legal_mask = final_map_bool | legal_mask_up | legal_mask_down | legal_mask_left | legal_mask_right
            road_color_rgb_lst_strict = [
                (216, 220, 224),  # 白色道路两边
                (224, 228, 232),
                (228, 232, 232),
                (224, 228, 232),
                (224, 224, 228),
                (236, 236, 240),
                (232, 236, 236),
                (234, 234, 237),
                (222, 226, 230),
                (218, 218, 222),
                (230, 230, 234),
                (221, 225, 229),
                (229, 229, 233),
                (241, 243, 244),
                (249, 186, 48), # 橙色道路边缘
                (249, 201, 94),
                (249, 230, 188),
                (247, 216, 135),
                (247, 212, 115),
                (247, 208, 99),
                (244, 226, 171),
                (248, 210, 115),
                (244, 211, 109),
                (244, 219, 153),
                (248, 207, 105)
                
            ]
            for road_color in road_color_rgb_lst_strict:
                road_map = roughly_equal_bool(img, road_color, diff_limit=5)
                final_map_bool = final_map_bool | road_map 
            final_map_bool = final_map_bool & legal_mask

            # 增加background颜色：分割两条相邻道路
            bi_img_bg = np.zeros(img.shape[:2], dtype=np.bool)
            bg_color_rgb_lst = [
                (242, 243, 244),
                (248, 249, 250)
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
                '24.9760994936954,55.1568603515625',
                '24.89640226655871,67.137451171875',
                '24.63203814959688,46.8017578125',
                
            ]
            do = False
            for do_name in do_lst:
                if do_name in img_path:
                    do = True
            if do:
                road_color_rgb_lst_strict = [
                    (241, 243, 244), # 灰色的道路
                    (237, 241, 241),
                    (225, 225, 229),
                    (233, 233, 237),
                    (221, 221, 225),
                    (226, 226, 230),
                    (227, 227, 227),
                    (236, 236, 236),
                    (224, 224, 228),
                    (242, 241, 240),
                    # (234, 238, 238), # 白点
                    (222, 222, 226),
                    (218, 218, 222),
                    (218, 222, 226),
                    (230, 230, 234),

                ]
                for road_color in road_color_rgb_lst_strict:
                    road_map = roughly_equal_bool(img, road_color, diff_limit=1)
                    final_map_bool = final_map_bool | road_map 
                final_map_bool = final_map_bool

            # 罕见的道路颜色
            do_lst = [
                '24.891419479211137,55.0689697265625'
            ]
            do = False
            for do_name in do_lst:
                if do_name in img_path:
                    do = True
            if do:
                road_color_rgb_lst_strict = [
                    (241, 243, 244), # 灰色的道路
                    (228, 232, 232),
                    (224, 224, 228),
                    (220, 220, 224),
                    (236, 236, 236),
                    (220, 220, 224),
                    
                ]
                for road_color in road_color_rgb_lst_strict:
                    road_map = roughly_equal_bool(img, road_color, diff_limit=1)
                    final_map_bool = final_map_bool | road_map 
                final_map_bool = final_map_bool                
            
            # 过滤小面积白色
            final_map = final_map_bool.astype(np.uint8)
            contours, hierarchy = cv2.findContours(final_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    cv2.drawContours(final_map, contour, contourIdx=-1, color=0, thickness=-1)
            final_map = final_map.astype(np.int) * 255
            
            # 去黑点
            ignore_black_lst = [
                '24.6370313535095,46.82373046875',
                '24.6370313535095,46.8951416015625',
                '24.39713301739104,54.51416015625',
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
            img_ = img.copy()
            img[final_map == 255] = (0, 0, 255)
            road_mask = (final_map == 255)
            color_img = (road_mask[:, :, None] * img_).astype(np.uint8)
            cv2.imwrite(img_path.replace('.png', '_color_vis.png'), color_img)
            cv2.imwrite(img_path.replace('.png', '_bi.png'), final_map)
            cv2.imwrite(img_path.replace('.png', '_vis.png'), img)    
       