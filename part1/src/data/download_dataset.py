import os
from collections import namedtuple
from time import sleep
from urllib.request import urlretrieve

import click
import tqdm

Image = namedtuple('Image', ['Name', 'URL'])


@click.command()
@click.argument('urls_path', type=click.File('r'),
                default=os.path.join('..', '..', 'resources', 'sample_files.csv'))
def get_urls(urls_path):
    return [Image(*i.strip().split(",")) for i in urls_path.readlines()]


@click.command()
@click.argument('output_path', type=click.Path('r'),
                default=os.path.join('..', '..', 'resources', 'sample_files.csv'))
@click.option('-d', '--delay', 'delay', default=2, type=int)
def download_images(images, output_path, delay):
    for image in tqdm.tqdm(images):
        output_image = os.path.join(output_path, image.Name)
        if not os.path.isfile(output_image):
            urlretrieve(image.URL, output_image)
            sleep(delay)


if __name__ == '__main__':
    images = get_urls()
    download_images(images)
