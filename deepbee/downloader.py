import os
from collections import namedtuple
from time import sleep
from urllib.request import urlretrieve

import click
import tqdm

Image = namedtuple('Image', ['Name', 'URL'])
default_output_path = os.path.join('resources', 'output', 'frames_dataset')
default_samples_path = os.path.join('resources', 'frames_dataset', 'sample.csv')
default_delay = 2


def get_urls(urls):
    return [Image(*i.strip().split(",")) for i in urls]


def download_images(img_urls, output_path, delay):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for image in tqdm.tqdm(img_urls):
        output_image = os.path.join(output_path, image.Name)
        if not os.path.isfile(output_image):
            try:
                urlretrieve(image.URL, output_image)
                sleep(delay)
            except Exception as e:
                print(e)


@click.command()
@click.option('-u', '--urls-path', 'urls_path', type=click.File('r'), default=default_samples_path)
@click.option('-o', '--output-path', 'output_path', type=str, default=default_output_path)
@click.option('-d', '--delay', 'delay', default=default_delay, type=int)
def main(urls_path, output_path, delay):
    urls = urls_path.readlines()
    img_urls = get_urls(urls)
    download_images(img_urls, output_path, delay)


if __name__ == '__main__':
    main()
