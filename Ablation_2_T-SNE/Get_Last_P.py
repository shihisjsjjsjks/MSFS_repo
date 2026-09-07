from PIL import Image
import os

def crop_and_merge_images(
    image1_path,
    image2_path,
    output_path,
    right_ratio=0.5,  # 保留右半部分的宽度比例 (0-1)
    height_ratio=1.0,  # 保留高度比例 (0-1)
    crop_from_bottom=True  # True:从下往上保留，False:从上往下保留
):
    """
    将两张图片的右半部分裁剪并拼接到一起
    
    参数:
    - image1_path: 第一张图片路径
    - image2_path: 第二张图片路径
    - output_path: 输出图片保存路径
    - right_ratio: 右半部分保留程度 (0-1)，如0.5表示保留右半边的50%
    - height_ratio: 高度保留程度 (0-1)，如0.8表示保留原图高度的80%
    - crop_from_bottom: True表示从下往上保留，False表示从上往下保留
    """
    
    try:
        # 1. 打开两张图片
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)
        
        # 2. 获取原始尺寸
        width1, height1 = img1.size
        width2, height2 = img2.size
        
        print(f"图片1尺寸: {width1}x{height1}")
        print(f"图片2尺寸: {width2}x{height2}")
        
        # 3. 计算裁剪区域
        # 计算右半部分的起始x坐标
        start_x1 = width1 - int(width1 * right_ratio)
        start_x2 = width2 - int(width2 * right_ratio)
        
        # 计算裁剪高度
        crop_height1 = int(height1 * height_ratio)
        crop_height2 = int(height2 * height_ratio)
        
        # 计算y坐标（根据crop_from_bottom参数决定从上还是从下开始）
        if crop_from_bottom:
            # 从下往上保留：y坐标从底部减去裁剪高度
            start_y1 = height1 - crop_height1
            start_y2 = height2 - crop_height2
        else:
            # 从上往下保留：y坐标从顶部开始
            start_y1 = 0
            start_y2 = 0
        
        # 4. 裁剪图片
        # 图片1裁剪区域（右半部分）
        crop_area1 = (start_x1, start_y1, width1, start_y1 + crop_height1)
        cropped_img1 = img1.crop(crop_area1)
        
        # 图片2裁剪区域（右半部分）
        crop_area2 = (start_x2, start_y2, width2, start_y2 + crop_height2)
        cropped_img2 = img2.crop(crop_area2)
        
        print(f"图片1裁剪区域: {crop_area1} -> 尺寸: {cropped_img1.size}")
        print(f"图片2裁剪区域: {crop_area2} -> 尺寸: {cropped_img2.size}")
        
        # 5. 确保两张图片高度一致（取较小的高度）
        final_height = min(cropped_img1.height, cropped_img2.height)
        
        # 如果高度不一致，调整图片大小
        if cropped_img1.height != final_height:
            cropped_img1 = cropped_img1.resize(
                (cropped_img1.width, final_height), 
                Image.Resampling.LANCZOS
            )
        if cropped_img2.height != final_height:
            cropped_img2 = cropped_img2.resize(
                (cropped_img2.width, final_height), 
                Image.Resampling.LANCZOS
            )
        
        # 6. 创建新图片（宽度为两张裁剪图片之和）
        new_width = cropped_img1.width + cropped_img2.width
        new_image = Image.new('RGB', (new_width, final_height))
        
        # 7. 拼接图片（图片1在左，图片2在右）
        new_image.paste(cropped_img1, (0, 0))
        new_image.paste(cropped_img2, (cropped_img1.width, 0))
        
        # 8. 保存图片
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        new_image.save(output_path)
        
        print(f"图片已保存到: {output_path}")
        print(f"最终合并图片尺寸: {new_width}x{final_height}")
        
        # 关闭图片
        img1.close()
        img2.close()
        
        return True
        
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    # # 示例1：使用默认参数（保留右半部分，全高度）
    # crop_and_merge_images(
    #     image1_path="path/to/image1.jpg",
    #     image2_path="path/to/image2.jpg",
    #     output_path="output/merged_default.jpg"
    # )
    
    # 示例2：自定义裁剪参数
    crop_and_merge_images(
        image1_path="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/FT_Origin_test_6.png",
        image2_path="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/FT_Origin_anti_6.png",
        output_path="/root/autodl-tmp/ZLJ/Ablation_Pro/Ablation_2_T-SNE/XW/Origin.png",
        right_ratio=0.545,      # 只保留右半部分的30%
        height_ratio=0.86,     # 保留原图高度的80%
        crop_from_bottom=True  # 从上往下裁剪
    )
    
    # # 示例3：保留更多右半部分
    # crop_and_merge_images(
    #     image1_path="path/to/image1.jpg",
    #     image2_path="path/to/image2.jpg",
    #     output_path="output/merged_right70.jpg",
    #     right_ratio=0.7,      # 保留右半部分的70%（几乎整张图）
    #     height_ratio=0.9      # 保留原图高度的90%
    # )