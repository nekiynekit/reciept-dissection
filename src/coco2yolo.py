import cv2
import json

from tqdm import tqdm

import click

class ConvertCOCOToYOLO:

    """
    Takes in the path to COCO annotations and outputs YOLO annotations in multiple .txt files.
    COCO annotation are to be JSON formart as follows:

        "annotations":{
            "area":2304645,
            "id":1,
            "image_id":10,
            "category_id":4,
            "bbox":[
                0::704
                1:620
                2:1401
                3:1645
            ]
        }
        
    """

    def __init__(self, img_folder, json_path):
        self.img_folder = img_folder
        self.json_path = json_path
        

    def get_img_shape(self, img_path):
        img = cv2.imread(img_path)
        try:
            return img.shape
        except AttributeError:
            print('error!', img_path)
            return (None, None, None)

    def convert_labels(self, img_path, x1, y1, x2, y2):
        """
        Definition: Parses label files to extract label and bounding box
        coordinates. Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
        """

        def sorting(l1, l2):
            if l1 > l2:
                lmax, lmin = l1, l2
                return lmax, lmin
            else:
                lmax, lmin = l2, l1
                return lmax, lmin

        size = self.get_img_shape(img_path)
        xmax, xmin = sorting(x1, x2)
        ymax, ymin = sorting(y1, y2)
        dw = 1./size[1]
        dh = 1./size[0]
        x = (xmin + xmax)/2.0
        y = (ymin + ymax)/2.0
        w = xmax - xmin
        h = ymax - ymin
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        assert all([item >= 0. and item <= 1. for item in [x, y, w, h]]), f"{[size[1], size[0], x1, y1, x2, y2]}, img={img_path}"
        return (x,y,w,h)

    def convert(self):
        # Enter directory to read JSON file
        data = json.load(open(self.json_path))
        id_to_filename = {int(data["images"][i]["id"]): data["images"][i]["file_name"] for i in range(len(data["images"]))}

        
        check_set = set()

        # Retrieve data
        for i in tqdm(range(len(data['annotations']))):

            # Get required data
            image_id = str({data['annotations'][i]['image_id']})
            category_id = str(data['annotations'][i]['category_id'] - 1)
            bbox = data['annotations'][i]['bbox']
            img_path = id_to_filename[data['annotations'][i]['image_id']][:-4]

            # Retrieve image.
            if self.img_folder == None:
                image_path = f'{image_id}.jpg'
            else:
                image_path = f'./{self.img_folder}/{img_path}.jpg'


            # Convert the data
            kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            yolo_bbox = self.convert_labels(image_path, kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3])
            
            # Prepare for export
            
            filename = f'./{self.img_folder}/{img_path}.txt'
            content = f"{category_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}"

            # Export 
            if image_id in check_set:
                # Append to existing file as there can be more than one label in each image
                file = open(filename, "a")
                file.write("\n")
                file.write(content)
                file.close()

            elif image_id not in check_set:
                check_set.add(image_id)
                # Write files
                file = open(filename, "w")
                file.write(content)
                file.close()

@click.command()
@click.option("--img-folder", "-i", help="Path to images")
@click.option("--json-path", "-j", help="JSON file with annotations")
def main(img_folder, json_path):
    ConvertCOCOToYOLO(img_folder=img_folder,json_path=json_path).convert()


# To run in as a class
if __name__ == "__main__":
    main()
