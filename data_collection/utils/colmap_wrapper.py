import os
import subprocess

# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense
def run_colmap(basedir, match_type, cam_model, cam_params, call_colmap, dense_reconstruct=False):

    print(cam_params)
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')

    if dense_reconstruct:
        automatic_reonstructor_args = [
            call_colmap, 'automatic_reconstructor',
            '--workspace_path', basedir,
            '--image_path', os.path.join(basedir, 'images'),
            '--single_camera', '1',
            '--camera_model', cam_model
        ]

        auto_reconstruct_output = (subprocess.check_output(
                                        automatic_reonstructor_args, universal_newlines=True) )
        logfile.write(auto_reconstruct_output)
        print('Automatic reconstruction done')
    
    else:

        feature_extractor_args = [
            call_colmap, 'feature_extractor',
                '--database_path', os.path.join(basedir, 'database.db'),
                '--image_path', os.path.join(basedir, 'images'),
                '--ImageReader.single_camera', '1',
                '--ImageReader.camera_model', cam_model,
                '--ImageReader.camera_params', cam_params
                # '--SiftExtraction.use_gpu', '0',
        ]
        feat_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True) )
        logfile.write(feat_output)
        print('Features extracted')

        exhaustive_matcher_args = [
            call_colmap, match_type,
                '--database_path', os.path.join(basedir, 'database.db'),
        ]

        match_output = (subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
        logfile.write(match_output)
        print('Features matched')

        p = os.path.join(basedir, 'sparse')
        if not os.path.exists(p):
            os.makedirs(p)

        # mapper_args = [
        #     'colmap', 'mapper',
        #         '--database_path', os.path.join(basedir, 'database.db'),
        #         '--image_path', os.path.join(basedir, 'images'),
        #         '--output_path', os.path.join(basedir, 'sparse'),
        #         '--Mapper.num_threads', '16',
        #         '--Mapper.init_min_tri_angle', '4',
        # ]

        mapper_args = [
            call_colmap, 'mapper',
                '--database_path', os.path.join(basedir, 'database.db'),
                '--image_path', os.path.join(basedir, 'images'),
                '--output_path', os.path.join(basedir, 'sparse'), # --export_path changed to --output_path in colmap 3.6
                '--Mapper.num_threads', '36',
                '--Mapper.init_min_tri_angle', '4',
                '--Mapper.multiple_models', '0',
                '--Mapper.extract_colors', '0',
                '--Mapper.ba_refine_focal_length', '0',
                '--Mapper.ba_refine_extra_params', '0'
                # '--Mapper.abs_pose_min_num_inliers', '2'
        ]

        map_output = (subprocess.check_output(mapper_args, universal_newlines=True) )
        logfile.write(map_output)

    
    logfile.close()

    converter_args = [
        call_colmap, 'model_converter',
            '--input_path', os.path.join(basedir, 'sparse', '0'),
            '--output_path', os.path.join(basedir, 'sparse', '0'),
            '--output_type', 'TXT'
    ]

    converter_output = (subprocess.check_output(converter_args, universal_newlines=True) )

    print('Sparse map created')
    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )