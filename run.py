import src.ATL03_API as is2
import src.Depth_profile as dp
import src.Tide_API as ta
import src.Pixel_transformation as pt
import src.Sentinel_API as sa
import src.Coral_Reef as coral_reef

import json

def run_pipeline():
    params_f = open('config/data-params.json')
    params = json.load(params_f)

    reef_name = params['reef_name']
    data_dir = params['data_dir']
    start_date = params['sentinel_start_date']
    end_date = params['sentinel_end_date']
    num_imgs = params['num_sentinel_imgs']
    redownload_is2 = params['redownload_is2']
    download_sentinel = params['download_sentinel']
    earthdata_login = params['earthdata_login']
    earthdata_password = params['earthdata_password']
    sentinel_username = params['sentinel_username']
    sentinel_password = params['sentinel_password']
    params_f.close()
    print(reef_name)
    reef = coral_reef.Coral_Reef(data_dir, reef_name)
    if redownload_is2:
        is2.main(data_dir, reef_name,earthdata_login,earthdata_password)
        dp.get_depths(reef)
    if download_sentinel:
        sa.get_sentinel_images(reef, start_date, end_date, num_imgs,sentinel_username, sentinel_password)
    pt.all_safe_files(reef)



if __name__ == '__main__':
    run_pipeline()
