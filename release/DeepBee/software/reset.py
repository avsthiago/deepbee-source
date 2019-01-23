import os


PATH_PREDS = ['../annotations/predictions', '../annotations/predictions_corrected',
              '../annotations/detections', '../original_images', '../output/labeled_images', '../output/spreadsheet']

def main():
    print('This action will remove all predictions and images.\nIs it what you want?[Y,n]')
    answer = input()
    if answer == 'Y':
        for i in  PATH_PREDS:
            files = [os.path.join(i, a) for a in os.listdir(i) if os.path.isfile(os.path.join(i, a))]
            for j in files:
                os.remove(j)

if __name__ == '__main__':
    main()
    