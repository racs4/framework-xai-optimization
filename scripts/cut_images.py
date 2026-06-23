import os
from PIL import Image

def cut_images_and_save(
    images: list,
    output_dir: str,
    prefix: str,
    suffix: str,
    region: list
):
    os.makedirs(output_dir, exist_ok=True)

    for i, image in enumerate(images):
        # Crop the image
        cropped_image = image.crop(region)
        
        # Save the cropped image
        image_filename = f"{prefix}_{i}_{suffix}.png"
        image_path = os.path.join(output_dir, image_filename)
        cropped_image.save(image_path)
        
                
    print("All images and masks have been saved.")


if __name__ == "__main__":
    # Example usage
    images = [
        Image.open("C:\\Users\\JARVIS\\Documents\\mestrado-rheidner\\otimizacao-2\\results\\results_imagenette_clip\\337\\literature\\heatmap_guided_backprop.png"), 
        Image.open("C:\\Users\\JARVIS\\Documents\\mestrado-rheidner\\otimizacao-2\\results\\results_imagenette_clip\\337\\literature\\heatmap_guided_gradcam.png"),
        Image.open("C:\\Users\\JARVIS\\Documents\\mestrado-rheidner\\otimizacao-2\\results\\results_imagenette_clip\\337\\literature\\heatmap_saliency.png"),
        Image.open("C:\\Users\\JARVIS\\Documents\\mestrado-rheidner\\otimizacao-2\\results\\results_imagenette_clip\\337\\literature\\heatmap_integrated_gradients.png"),
        Image.open("C:\\Users\\JARVIS\\Documents\\mestrado-rheidner\\otimizacao-2\\results\\results_imagenette_clip\\337\\literature\\heatmap_layer_gradcam.png"),
        Image.open("C:\\Users\\JARVIS\\Documents\\mestrado-rheidner\\otimizacao-2\\results\\results_imagenette_clip\\337\\ipha_flavio_marcelo_v\\heatmap_quantity.png"),
        Image.open("C:\\Users\\JARVIS\\Documents\\mestrado-rheidner\\otimizacao-2\\results\\results_imagenette_clip\\337\\menor_probabilidade_menor_area_inpaint_v\\heatmap_quantity.png"),
        Image.open("C:\\Users\\JARVIS\\Documents\\mestrado-rheidner\\otimizacao-2\\results\\results_imagenette_clip\\337\\menor_probabilidade_menor_area_v\\heatmap_quantity.png"),
        Image.open("C:\\Users\\JARVIS\\Documents\\mestrado-rheidner\\otimizacao-2\\results\\results_imagenette_clip\\337\\poligono_e_perimetro_com_inpaint_v\\heatmap_quantity.png"),
        Image.open("C:\\Users\\JARVIS\\Documents\\mestrado-rheidner\\otimizacao-2\\results\\results_imagenette_clip\\337\\poligono_e_perimetro_v\\heatmap_quantity.png"),
    ]
    output_dir = "output_images"
    prefix = "image"
    suffix = "cropped"
    region = (158,60,374,446)  # Define your region here

    cut_images_and_save(images, output_dir, prefix, suffix, region)