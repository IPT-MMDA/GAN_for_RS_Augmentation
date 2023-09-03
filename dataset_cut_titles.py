import os
import numpy as np
from tqdm import trange
from osgeo import gdal

os.system("gdalbuildvrt -o out_temp.vrt s*.tif")
os.system("gdalbuildvrt -b 1 -o out_channel_1.vrt out_temp.vrt")
os.system("gdalbuildvrt -b 2 -o out_channel_2.vrt out_temp.vrt")
os.system("gdalbuildvrt -b 3 -o out_channel_3.vrt out_temp.vrt")
os.system("gdalbuildvrt -b 4 -o out_channel_4.vrt out_temp.vrt")
os.system("gdalbuildvrt -separate -o out_final.vrt out_channel_*.vrt kyiv_map_2021.tif")

ds = gdal.Open("out_final.vrt")


def save_file(input_file, output_file, index_x, index_y):
    ds = gdal.Open(input_file)
    array = ds.GetRasterBand(5).ReadAsArray(128 * index_x + 1, 128 * index_y + 1, 128, 128)

    array = array.astype(int)
    if np.sum(array == 0) > 0.1 * array.size:
        return False

    # python function doesn't work right
    # ds = gdal.Translate(output_file, ds, srcWin=[128*index_x+1, 128*index_y+1, 128, 128])
    coords = "{} {} {} {}".format(128 * index_x + 1, 128 * index_y + 1, 128, 128)
    os.system("gdal_translate -srcwin {} {} {}".format(coords, input_file, output_file))
    return True


out_dir = "titles"
dropped, written = 0, 0
for index_x in trange(ds.RasterXSize // 128 - 1):
    for index_y in range(ds.RasterYSize // 128 - 1):
        out_title_name = os.path.join("cropped_256x256", "title_{}_{}.tiff".format(index_x, index_y))
        success = save_file("out_final.vrt", out_title_name, index_x, index_y)
        if success:
            written += 1
        else:
            dropped += 1
print("Written titles:", written)
print("Dropped titles:", dropped)
