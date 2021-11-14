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
    # img_dir = r'16zoom-11.02/OSM/osm_18'
    img_dir = r'temp'
    min_area = 16

    img_paths = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if '_bi' in file or 'vis' in file or 'color' in file or '.xlsx' in file:
                continue
            if not '-o-' in file:
                continue
            # if '_' in file:
            #     continue            
            img_paths.append(os.path.join(root, file))

    # img_paths = [r'16zoom-11.02/OSM_512\29.35345166863501-48.0157470703125-16-USELESS-USELESS-512-USELESS-o-lbl0.png']
    for img_i, img_path in enumerate(tqdm.tqdm(img_paths)):
        # if img_i > 50:
        #     break
        if True:
            
              
            img = cv2.imread(img_path)
            bi_img = np.zeros(img.shape[:2], dtype=np.bool)
            # road颜色
            road_color_rgb_lst = [
                # (255, 255, 255), # 白色道路中间
                # (254, 253, 215), # 黄色道路
                # (253, 235, 206),
                # (255, 237, 193),
                # (254, 240, 205),
                # (255, 233, 165), # 橙色道路
                # (251, 219, 152),
                # new color value
                (255, 255, 255), # 白色道路中间
                (249, 178, 156), # 橙色道路
                (252, 214, 164), # 土黄色道路
                (247, 250, 191), # 黄绿色道路
                (232, 146, 162), # 玫红色道路
            ]
            for road_color in road_color_rgb_lst:
                limit = 40
                road_map = roughly_equal_bool(img, road_color, diff_limit=limit)
                bi_img = bi_img | road_map


            bi_img = bi_img.astype(np.int) * 255

            # background颜色
            # 
            bi_img_bg = np.zeros(img.shape[:2], dtype=np.bool)
            bg_color_rgb_lst = [
                (255, 255, 229),

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

            ]
            for road_color in road_color_rgb_lst_strict:
                road_map = roughly_equal_bool(img, road_color, diff_limit=5)
                final_map_bool = final_map_bool | road_map 
            final_map_bool = final_map_bool & legal_mask



            # 过滤小面积白色
            final_map = final_map_bool.astype(np.uint8)
            # contours, hierarchy = cv2.findContours(final_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # for contour in contours:
            #     area = cv2.contourArea(contour)
            #     if area < min_area:
            #         cv2.drawContours(final_map, contour, contourIdx=-1, color=0, thickness=-1)
            final_map = final_map.astype(np.int) * 255
            
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
            black2white_mask = sum_map > 4
            self_black_map = final_map_mask.astype(np.bool)
            black2white_mask = (black2white_mask & ~self_black_map) | self_black_map
            after_map = black2white_mask.astype(np.int) * 255
            final_map = after_map            
            
            # 过滤小面积白色
            final_map = (final_map / 255).astype(np.uint8)
            contours, hierarchy = cv2.findContours(final_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    cv2.drawContours(final_map, contour, contourIdx=-1, color=0, thickness=-1)
            final_map = (final_map * 255).astype(np.uint8)
            #
            img_ = img.copy()
            img[final_map == 255] = (0, 0, 255)
            road_mask = (final_map == 255)
            color_img = (road_mask[:, :, None] * img_).astype(np.uint8)
            cv2.imwrite(img_path.replace('.png', '_color_vis.png'), color_img)
            kernel = np.ones((2, 2), dtype=np.uint8)
            final_map = cv2.morphologyEx(final_map.astype(np.uint8), cv2.MORPH_OPEN, kernel, 1)
            cv2.imwrite(img_path.replace('.png', '_bi.png'), final_map)
            cv2.imwrite(img_path.replace('.png', '_vis.png'), img)  