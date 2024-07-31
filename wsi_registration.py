from valis import registration

slide_src_dir = "/path/to/slides"
results_dst_dir = "/path/to/output/"
registered_slide_dst_dir = "/path/to/registered/slides"
reference_slide = "wsi_ref.tiff"

def register(size=1024):
    registrar = registration.Valis( slide_src_dir, 
                                    results_dst_dir, 
                                    reference_img_f='HE_ref.tiff',
                                    thumbnail_size=size, 
                                    max_image_dim_px=size,
                                    max_processed_image_dim_px=size,
                                    max_non_rigid_registration_dim_px=size,
                                  )
    registrar.register()
    registrar.warp_and_save_slides(registered_slide_dst_dir)
    registration.kill_jvm()

