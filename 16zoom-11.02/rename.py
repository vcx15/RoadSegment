import os
import shutil
import tqdm

#img_dir_lst = [r'16zoom\Bing_512', r'16zoom\Google_1024', r'16zoom\OSM_512']
img_dir_lst = [ r'16zoom-11.02/Bing/bing_18']


for img_dir in img_dir_lst:
    img_paths = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if '_bing_' not in file:
                if '_bi' in file or 'vis' in file or 'color' in file or '.xlsx' in file:
                    img_paths.append(os.path.join(root, file))
            else:
                if '_bi_' in file or 'vis' in file or 'color' in file or '.xlsx' in file or 'bi.png' in file:
                    img_paths.append(os.path.join(root, file))            

    for path in tqdm.tqdm(img_paths):
        os.remove(path)

    map_type = 'b' if 'Bing' in img_dir else 'g' if 'Google' in img_dir else 'o'
    resolution = 512 if '512' in img_dir else 1024
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            lat = file.split(',')[0]
            long = file.split('_')[0].split(', ')[1]
            zoom = file.split('_')[1]
            new_name = '{}-{}-{}-{}-{}-{}-{}-{}-lbl0.png'.format(
                lat, long, zoom, 'USELESS', 'USELESS', resolution, 'USELESS', map_type
            )
            new_path = os.path.join(root, new_name)
            old_path = os.path.join(root, file)
            os.rename(old_path, new_path)


