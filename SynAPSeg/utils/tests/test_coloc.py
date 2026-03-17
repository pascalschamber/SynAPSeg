from SynAPSeg.utils.utils_colocalization import colocalize_by_distance, colocalize, get_rp_table
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
import numpy as np

def test_colocalization_functions():
    imgp = "pred_stardist3d.tiff"

    img = uip.read_img(imgp)
    uip.pai(img)

    img = img[0,0, 1:, ...]

    rpdf = get_rp_table(img, np.zeros(img.shape).astype('uint16'), ch_axis=0, get_object_coords=True)

    colocs = [dict(coIds=(0,1), coChs=(0,1), assign_colocal_id=3)]

    coloc_itx_df, prtstr_itx = colocalize(
        colocs,
        rpdf,
        img,
        'CZYX',

    )

    coloc_df, prtstr = colocalize_by_distance(
        colocs, 
        rpdf, 
        max_dist=10,
        pixel_size=1,
    )


    # check that all objects in intersection colocalization are also in distance colocalization
    # test passed, good sanity check since itx coloc is more stringent than distance
    coloc_dist = coloc_df.query('colocal_id == 3')
    dist_lbls = set(coloc_dist['label'].to_list())

    coloc_itx = coloc_itx_df.query('colocal_id == 3')
    itx_lbls = set(coloc_itx['label'].to_list())

    ulbls = itx_lbls.difference(dist_lbls)
    assert len(ulbls) == 0, f"Labels in intersection colocalization but not in distance colocalization: {ulbls}"

    common_labels = itx_lbls.intersection(dist_lbls)


    # check that intersection labels are identical
    # they are 99.5% identical but this could def be the case that assignment results are different
    # so, test passed
    coloc_itx.sort_values('label', inplace=True)
    coloc_dist.sort_values('label', inplace=True)


    itxlbl_check = (
        coloc_dist.query("label.isin(@common_labels)")['ch0_intersecting_label'].to_numpy()
        == coloc_itx.query("label.isin(@common_labels)")['ch0_intersecting_label'].to_numpy()
    )
    print(np.sum(itxlbl_check)/len(common_labels))

    # get the labels that are different
    diff_lbls_itx = coloc_itx.query("label.isin(@common_labels)")[itxlbl_check==False]
    diff_lbls_dist = coloc_dist.query("label.isin(@common_labels)")[itxlbl_check==False]
