import os

map_color_width_info = {
    'google': { # for google 1024, z16
        "name": ['orange_40', 'white_22', 'white_10', 'white_40', 'white_60', 'grey_20'],
        "width": [40, 22, 10, 40, 60, 20],
        "color": ['orange', 'white', 'white', 'white', 'white', 'grey'],  # rgb
        "color_name": {'grey':(213, 216, 219), 'white':(255, 255, 255), 'orange':(253, 226, 147)},
        "name_map": {
            'grey_20':'railway', 'orange_40':'freeway', 'white_40': 'primary road', 'white_22': 'secondary road',
            'white_10':'residential road', 'white_60':'trunk'
        }
    },
    'bing': { # for bing 512, z16
        "name": ['orange_11', 'yellow_7', 'purple_12', 'white_23', 'skin_10', 'white_10'],
        "width": [11, 7, 12, 23, 10, 10],
        "color": ['orange', 'yellow', 'purple', 'white', 'skin', 'white'],  # rgb
        "color_name": {'orange':(255, 244, 171), 'yellow':(255, 254, 237), 'white':(255, 255, 255), 'purple':(233, 211, 250), 'skin':(247, 242, 229)},
        "name_map": {
            'skin_10':'secondary road', 'orange_11':'trunk', 'yellow_7':'primary road', 
            'purple_12':'freeway', 'white_23':'secondary road', 'white_10':'residential road'
        }
    },  
    'osm': { # for osm 512, z16
        "name": ['yellow_20', 'white_16', 'white_6', 'white_10', 'grey_10', 'orange_20'],
        "width": [20, 16, 6, 10, 10, 20],
        "color": ['yellow', 'white', 'white', 'white', 'grey', 'orange'],  # rgb
        "color_name": {'yellow':(254, 253, 215), 'white':(255, 255, 255), 'orange':(255, 233, 165),
                        'grey':(234, 233, 230)},
        "name_map": {
          'grey_10':'railway', 'yellow_20':'trunk', 'white_16':'primary road', 'white_6':'residential road',
          'white_10':'secondary road', 'orange_20':'freeway'
        }        
    }       
}