import click
import cv2
import numpy as np


@click.command()
@click.option("--imgpath", "-i", default="examples/reciept.jpg")
def main(imgpath):
    img = cv2.imread(imgpath)
    cv2.imshow("recipent image", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
