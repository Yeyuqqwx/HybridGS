import os
os.chdir(os.path.dirname(__file__))
from args import args, options
import torch
from scene import Scene
import json
from gs3d.gaussian_renderer import render
from tqdm import tqdm
from gs3d.gaussian_renderer import GaussianModel
from utils.image_utils import psnr, ssim
import lpips
from PIL import Image, ImageDraw, ImageFont

lpips_fn = lpips.LPIPS(net='vgg').cuda()
gaussians = GaussianModel(args.gs3d.sh_degree, args.gs3d.dim_embed, args.gs3d.embed)
scene = Scene(args.model_path, args.dataset, gaussians, create_gs_now=False)

test_only = False
save_cat = True
save_split = True

if options.metric_only == 'True':
    test_only = True
    save_cat = False
    save_split = False

save_path = options.ckpt_path
print(f'Saving to {save_path}')
torch.cuda.empty_cache()
scene.load(ckpt_path=options.ckpt_path)

if args.gs3d.dim_embed > 0:
    gaussians.embedding.eval()
bg_color = [1, 1, 1] if args.dataset.white_background else [0, 0, 0]
bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
modes = ["test"] if test_only else ["train", "test"]
for mode in modes:
    os.makedirs(os.path.join(save_path, mode), exist_ok=True)
    if save_cat:
        cat_path = os.path.join(save_path, mode, "cat")
        os.makedirs(cat_path, exist_ok=True)
    if save_split:
        r_s_path = os.path.join(save_path, mode, "r_s")
        os.makedirs(r_s_path, exist_ok=True)
        gt_path = os.path.join(save_path, mode, "gt")
        os.makedirs(gt_path, exist_ok=True)
        if mode == 'train':
            r_st_path = os.path.join(save_path, mode, "r_st")
            os.makedirs(r_st_path, exist_ok=True)
            r_t_path = os.path.join(save_path, mode, "r_t")
            os.makedirs(r_t_path, exist_ok=True)
            mask_t_path = os.path.join(save_path, mode, "mask_t")
            os.makedirs(mask_t_path, exist_ok=True)
    per_view_dict = {}
    ssims_rs_GT = []
    psnrs_rs_GT = []
    lpipss_rs_GT = []
    ssims_r_GT = []
    psnrs_r_GT = []
    lpipss_r_GT = []
    if mode == "test":
        views = scene.getTestCameras()
        args.gs3d.embed = ''
        gaussians.embed = ''
    else:
        views = scene.getTrainCameras()
        args.gs3d.embed = 'gaussianimage'
        gaussians.embed = 'gaussianimage'
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        image_name = view.image_name

        gt = view.original_image[0:3, :, :]
        render_pkg = render(view, gaussians, args.pipe, bg)
        r_s = render_pkg["render"]
        r_s = torch.clamp(r_s, 0, 1)
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]
        if gaussians.embed == 'gaussianimage':
            embed = gaussians.get_embed(torch.tensor(view.trainuid, device='cuda'))
            r_t, mask_t = gaussians.gaussianimage(embed, (r_s.shape[1], r_s.shape[2]), return_mask=True)
            r_t = torch.clamp(r_t, 0, 1)
            mask_t = torch.clamp(mask_t, 0, 1)
            r_st = r_t + (1 - mask_t) * r_s
            r_st = torch.clamp(r_st, 0, 1)

        ssim_rs_GT = ssim(r_s, gt).item()
        psnr_rs_GT = psnr(r_s.unsqueeze(0), gt.unsqueeze(0)).item()
        lpips_rs_GT = lpips_fn(r_s, gt).item()
        ssims_rs_GT.append(ssim_rs_GT)
        psnrs_rs_GT.append(psnr_rs_GT)
        lpipss_rs_GT.append(lpips_rs_GT)
        if args.gs3d.embed != '':
            ssim_r_GT = ssim(r_st, gt).item()
            psnr_r_GT = psnr(r_st.unsqueeze(0), gt.unsqueeze(0)).item()
            lpips_r_GT = lpips_fn(r_st, gt).item()
            ssims_r_GT.append(ssim_r_GT)
            psnrs_r_GT.append(psnr_r_GT)
            lpipss_r_GT.append(lpips_r_GT)

        if save_cat:
            font_size = 30
            font_color = (0, 255, 0)
            font = ImageFont.load_default(size=font_size)
            if args.gs3d.embed in ['gaussianimage']:
                save_img = torch.cat([gt, r_st.squeeze(0).squeeze(0), r_s.squeeze(0), r_t.squeeze(0), mask_t.squeeze(0)], dim=1) \
                    .mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            else:
                save_img = torch.cat([gt, r_s], dim=1) \
                    .mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            save_img = Image.fromarray(save_img)
            draw = ImageDraw.Draw(save_img)
            if args.gs3d.embed == '':
                draw.text((10, 10 + gt.shape[1]), f"ssim: {ssim_rs_GT:.4f}\n"
                                                    f"psnr: {psnr_rs_GT:.4f}\n"
                                                    f"lpips: {lpips_rs_GT:.4f}", fill=font_color, font=font)
            else:
                draw.text((10, 10 + gt.shape[1]), f"ssim: {ssim_r_GT:.4f}\n"
                                                    f"psnr: {psnr_r_GT:.4f}\n"
                                                    f"lpips: {lpips_r_GT:.4f}", fill=font_color, font=font)
                draw.text((10, 10 + gt.shape[1] * 2), f"ssim: {ssim_rs_GT:.4f}\n"
                                                        f"psnr: {psnr_rs_GT:.4f}\n"
                                                        f"lpips: {lpips_rs_GT:.4f}", fill=font_color, font=font)

            save_img.save(os.path.join(cat_path, image_name))
        if save_split:
            save_gt = Image.fromarray(gt.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
            save_gt.save(os.path.join(gt_path, image_name))

            save_r_s = Image.fromarray(r_s.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
            save_r_s.save(os.path.join(r_s_path, image_name))
            if mode == 'train':
                save_r_st = Image.fromarray(r_st.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
                save_r_st.save(os.path.join(r_st_path, image_name))
                
                save_r_t = Image.fromarray(r_t.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
                save_r_t.save(os.path.join(r_t_path, image_name))

                save_mask_t = Image.fromarray(mask_t.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
                save_mask_t.save(os.path.join(mask_t_path, image_name))

        if args.gs3d.embed != '':
            per_view_dict[image_name] = {
                "ssim_r_s_GT": ssim_rs_GT,
                "psnr_r_s_GT": psnr_rs_GT,
                "lpips_r_s_GT": lpips_rs_GT,
                "ssim_r_GT": ssim_r_GT,
                "psnr_r_GT": psnr_r_GT,
                "lpips_r_GT": lpips_r_GT,
            }
        else:
            per_view_dict[image_name] = {
                "ssim_r_s_GT": ssim_rs_GT,
                "psnr_r_s_GT": psnr_rs_GT,
                "lpips_r_s_GT": lpips_rs_GT}

    with open(os.path.join(save_path, mode, "per_view_count.json"),
                'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    with open(os.path.join(save_path, mode, "metrics.txt"), 'w') as fp:
        if mode == 'train':
            print(f"r_GT:\n"
                    f"  SSIM : {torch.tensor(ssims_r_GT).mean():>12.7f}\n"
                    f"  PSNR : {torch.tensor(psnrs_r_GT).mean():>12.7f}\n"
                    f"  LPIPS: {torch.tensor(lpipss_r_GT).mean():>12.7f}\n"
                    f"rs_GT:\n"
                    f"  SSIM : {torch.tensor(ssims_rs_GT).mean():>12.7f}\n"
                    f"  PSNR : {torch.tensor(psnrs_rs_GT).mean():>12.7f}\n"
                    f"  LPIPS: {torch.tensor(lpipss_rs_GT).mean():>12.7f}\n"
                    , file=fp)
        else:
            print(f"rs_GT:\n"
                    f"  SSIM : {torch.tensor(ssims_rs_GT).mean():>12.7f}\n"
                    f"  PSNR : {torch.tensor(psnrs_rs_GT).mean():>12.7f}\n"
                    f"  LPIPS: {torch.tensor(lpipss_rs_GT).mean():>12.7f}\n"
                    , file=fp)
    
    if mode == 'train':
        print(f"r_GT: "
                f"{torch.tensor(ssims_r_GT).mean():.3f}, "
                f"{torch.tensor(psnrs_r_GT).mean():.3f}, "
                f"{torch.tensor(lpipss_r_GT).mean():.3f}\n"
                f"rs_GT: "
                f"{torch.tensor(ssims_rs_GT).mean():.3f}, "
                f"{torch.tensor(psnrs_rs_GT).mean():.3f}, "
                f"{torch.tensor(lpipss_rs_GT).mean():.3f}\n")
    else:
        print(
                f"{torch.tensor(ssims_rs_GT).mean():.3f}, "
                f"{torch.tensor(psnrs_rs_GT).mean():.3f}, "
                f"{torch.tensor(lpipss_rs_GT).mean():.3f}\n")