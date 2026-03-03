import os
import random
import shutil
def split_data(fewshot_dir, imgs, labels, classes, n_shot):
    os.makedirs(fewshot_dir, exist_ok=True)

    class_to_images = {cls: [] for cls in classes}
    for img_path, label in zip(imgs, labels):
        class_name = classes[label]
        class_to_images[class_name].append(img_path)

    for cls_name, img_list in class_to_images.items():
        cls_save_dir = os.path.join(fewshot_dir, cls_name.replace(" ", "_"))
        os.makedirs(cls_save_dir, exist_ok=True)

        chosen_imgs = random.sample(img_list, min(n_shot, len(img_list)))

        for img_path in chosen_imgs:
            filename = os.path.basename(img_path)
            dst_path = os.path.join(cls_save_dir, filename)
            if not os.path.exists(dst_path):
                shutil.copy(img_path, dst_path)