import os
from multiprocessing import Pool
from tqdm import tqdm
import gc
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from AdvancedImageAnalysis import AdvancedImageAnalysis


class ImageAnalyst:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.images_dict = dict()
        self.num_processed = 0

        self.advanced_analysis = AdvancedImageAnalysis()
    
    def check_data(self, filename):
        """Check data format is appropriate."""
        # if filename not in self.images_dict and filename.lower().endswith(
        #     (".png", ".jpg", ".jpeg")
        # ):
        #     return True
        return True
        
    def closest_save_interval(self, chunk_size, save_interval):
        assert chunk_size >= save_interval, 'Данные не сохраняется'
        for i in range(save_interval, 0, -1):
            if chunk_size % i == 0:
                return i
        return 1

    def analyse(
            self, 
            images_path,
            num_processes=4,
            compute_psnr=True,
            compute_brisque=True,
            compute_tv=True,
            compute_distances=True,
            save_interval=100
        ):
        """Run the image analysis in a multiprocessing environment."""
        if os.path.isdir(images_path):
            filenames = sorted(os.listdir(images_path))
            num_files = len(filenames)
            chunk_size = num_files // num_processes
            save_interval = self.closest_save_interval(chunk_size, save_interval)

            pool = Pool(num_processes)
            results = []
            for i in range(num_processes):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != num_processes - 1 else num_files
                sub_filenames = filenames[start_idx:end_idx]
                results.append(
                    pool.apply_async(
                        self._analyze_chunk,
                        args=(
                            images_path,
                            sub_filenames,
                            compute_psnr,
                            compute_brisque,
                            compute_tv,
                            compute_distances,
                            save_interval,
                        ),
                    )
                )
            pool.close()
            pool.join()
            for res in results:
                self.images_dict.update(res.get())

        elif os.path.isfile(images_path):
            if self.check_data(images_path):
                self._analyze_image(
                    images_path,
                    compute_psnr,
                    compute_brisque,
                    compute_tv,
                    compute_distances,
                )
        else:
            print(f"Invalid image path: {images_path}")

    def _analyze_chunk(
            self, 
            images_path,
            filenames,
            compute_psnr,
            compute_brisque,
            compute_tv,
            compute_distances,
            save_interval
        ):
        """Process a chunk of image paths."""
        chunk_dict = {}
        for filename in tqdm(filenames):
            if self.check_data(filename):
                filepath = os.path.join(images_path, filename)
                self._analyze_image(
                    filepath,
                    compute_psnr,
                    compute_brisque,
                    compute_tv,
                    compute_distances,
                )
                self.num_processed += 1
                if self.num_processed % save_interval == 0:
                    chunk_dict.update(self.images_dict)
                    self.save_csv()
        return chunk_dict

    def _analyze_image(
            self, 
            filepath,
            compute_psnr,
            compute_brisque,
            compute_tv,
            compute_distances,
        ):
        """Analyze an image using the specified metrics."""
        
        img_pairs = dict()
        for im in os.listdir(filepath):
            k = im.split("_")[-1].split(".")[0]
            img_pairs[k] = Image.open(os.path.join(filepath, im)).convert('RGB') 
        
        
        filepath = os.path.basename(filepath)
        
        self.images_dict[filepath] = dict()
        
        if compute_psnr:
            self.images_dict[filepath].update(self.advanced_analysis.psnr_metric(img_pairs.get("true"), 
                                                                                 img_pairs.get("generated")))
        if compute_brisque:
            self.images_dict[filepath].update(self.advanced_analysis.brisque_metric(img_pairs.get("generated")))
        # if compute_tv:
        #     self.images_dict[filepath].update(self.advanced_analysis.tv_metric(img_pairs.get("generated")))
        if compute_distances:
            log_dir = "/".join(self.log_file_path.split("/")[:-1])
            distances_dict = self.advanced_analysis.distance_metric(img_pairs.get("true"), img_pairs.get("generated"))
            if not os.path.exists(os.path.join(log_dir, "distances", filepath)): 
                os.makedirs(os.path.join(log_dir, "distances", filepath)) 
            for dist, res in distances_dict.items():
                res.savefig(f"{os.path.join(log_dir, 'distances', filepath, filepath)}_{dist}.png")
                plt.close(res)
        
    
    def save_csv(self):
        df = pd.DataFrame.from_dict(self.images_dict, orient="index")
        if os.path.exists(self.log_file_path):
            mode = 'a'
            header = False
        else:
            mode = 'w'
            header = True
        df.to_csv(self.log_file_path, mode=mode, header=header) # control header
        self.images_dict = {}
        gc.collect()