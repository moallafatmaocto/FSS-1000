import click

from src.data_preprocessing.config import TEST_DATA_PATH
from src.evaluation.predict import main
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


@click.command()
@click.option('--class-num', '-N', default=1, type=int, required=True)
@click.option('--sample-num-per_class', '-K', default=5, type=int, required=True)
@click.option('--batch-num-per_class', default=1, type=int)
@click.option('--model-save-path', type=str, default='relation_network_trained')
@click.option('--use-gpu', default=False, type=bool)
@click.option('--gpu', default=0, type=int)
@click.option("-td", "--test-dir", type=str, default=TEST_DATA_PATH)
@click.option("--result-dir", type=str, default='results_predicted_images_masks')
@click.option("--save-episode", type=int, default=9)
def entry_point_predict(class_num: int, sample_num_per_class: int, batch_num_per_class: int, model_save_path: str,
                        use_gpu: bool, gpu: int, test_dir: str, result_dir: str, save_episode: int):
    main(class_num, sample_num_per_class, batch_num_per_class, model_save_path,
         use_gpu, gpu, test_dir, result_dir, save_episode)


if __name__ == '__main__':
    entry_point_predict()
