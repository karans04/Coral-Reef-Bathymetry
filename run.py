import src.ATL03_API as is2
import src.Depth_profile as dp
import src.Tide_API as ta
import src.Pixel_transformation as pt

import json

def run_pipeline():
    params_f = open('config/data-params.json')
    params = json.load(params_f)

    reef_name = params['reef_name']
    data_dir = params['data_dir']

    params_f.close()

    # is2.main(data_dir, reef_name)
    # dp.get_depths(data_dir, reef_name)
    pt.all_safe_files(data_dir,reef_name)

if __name__ == '__main__':
    run_pipeline()
