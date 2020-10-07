import click

from src.evaluation.autolabel import main


@click.command()
@click.option('--class-num', '-N', default=1, type=int, required=True)
@click.option('--sample-num-per_class', '-K', default=5, type=int, required=True)
@click.option('--batch-num-per_class', default=1, type=int)
@click.option('--encoder-save-path', '-encoder', type=str, default='models/feature_encoder_pretrained_9_1_way_5shot.pkl')
@click.option('--network-save-path', '-network', type=str, default='models/relation_network_pretrained_9_1_way_5shot.pkl')
@click.option('--use-gpu', default=False, type=bool)
@click.option('--gpu', default=0, type=int)
@click.option("-sd", "--support-dir", type=str, default='test_data/african_elephant/supp')
@click.option("-td", "--test-dir", type=str, default='test_data/african_elephant/test')
@click.option("--result-dir", type=str, default='result')

def entry_point_test(class_num: int, sample_num_per_class: int, batch_num_per_class: int, encoder_save_path: str,
                     network_save_path: str,
                     use_gpu: bool, gpu: int, support_dir: str, test_dir: str,result_dir:str):
    main(class_num, sample_num_per_class, batch_num_per_class, encoder_save_path,
         network_save_path,
         use_gpu, gpu, support_dir, test_dir,result_dir)


if __name__ == '__main__':
    entry_point_test()
