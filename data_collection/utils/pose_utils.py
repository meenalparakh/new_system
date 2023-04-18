import numpy as np
import os
import sys
import imageio
import skimage.transform
import pickle
from data_collection.utils.colmap_wrapper import run_colmap
import data_collection.utils.colmap_read_model as read_model

def load_colmap_data(realdir):

    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])

    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    # w2c_mats = []
    poses = []
    image_fnames = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])

    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)

    for k in imdata:
        im = imdata[k]
        image_fnames.append(im.name)
        poses.append((im.tvec, im.qvec))
        # R = im.qvec2rotmat()
        # t = im.tvec.reshape([3,1])
        # m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        # w2c_mats.append(m)

    # w2c_mats = np.stack(w2c_mats, 0)
    return poses, image_fnames

    # c2w_mats = np.linalg.inv(w2c_mats)

    # poses = w2c_mats
    # poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    # poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)

    # points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    # pts3d = read_model.read_points3d_binary(points3dfile)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    # poses = np.concatenate([poses[:, 1:2, :],
    #                         poses[:, 0:1, :],
    #                         -poses[:, 2:3, :],
    #                         poses[:, 3:4, :],
    #                         poses[:, 4:5, :]],
    #                        1)

    return poses, pts3d, perm

def save_poses(basedir, poses, fnames):
    # with open(os.path.join(basedir, 'poses_bounds.npy'), 'wb'), poses)

    with open(os.path.join(basedir, 'image_poses.pkl'), 'wb') as f:
        pickle.dump((poses, fnames), f)

    return 

def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    with open(os.path.join(basedir, 'image_poses.pkl'), 'rb') as f:
        poses, fnames = pickle.load(f)

    return poses, fnames

def minify_v0(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    def downsample(imgs, f):
        sh = list(imgs.shape)
        sh = sh[:-3] + [sh[-3]//f, f, sh[-2]//f, f, sh[-1]]
        imgs = np.reshape(imgs, sh)
        imgs = np.mean(imgs, (-2, -4))
        return imgs

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgs = np.stack([imageio.imread(img)/255. for img in imgs], 0)

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)

        if isinstance(r, int):
            imgs_down = downsample(imgs, r)
        else:
            imgs_down = skimage.transform.resize(imgs, [imgs.shape[0], r[0], r[1], imgs.shape[-1]],
                                                order=1, mode='constant', cval=0, clip=True, preserve_range=False,
                                                 anti_aliasing=True, anti_aliasing_sigma=None)

        os.makedirs(imgdir)
        for i in range(imgs_down.shape[0]):
            imageio.imwrite(os.path.join(imgdir, 'image{:03d}.png'.format(i)), (255*imgs_down[i]).astype(np.uint8))


def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100./r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

def gen_poses(basedir, match_type, cam_model=1, cam_params=None,
            call_colmap='colmap', dense_reconstruct=False,
            factors=None):

    print(f'base_dir: {basedir}, \nmatch_type:{match_type}, \ncam_model={cam_model}, \ncam_params:{cam_params}')
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print( 'Need to run COLMAP' )
        run_colmap(basedir, match_type, cam_model, cam_params, call_colmap,
                    dense_reconstruct=dense_reconstruct)
    else:
        print('Don\'t need to run COLMAP')

    print( 'Post-colmap')

    # poses, pts3d, perm = load_colmap_data(basedir)
    poses, fnames = load_colmap_data(basedir)

    # save_poses(basedir, poses, pts3d, perm)
    save_poses(basedir, poses, fnames)

    # if factors is not None:
    #     print( 'Factors:', factors)
    #     minify(basedir, factors)

    print( 'Done with imgs2poses' )

    return True
