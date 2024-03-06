import os
from options.test_options import TestOptions
from dataloader.data_loader import dataloader_full
from model.models import create_model
from utils.evaluate import test_evaluate, get_dict_motion_category, get_save_path
import shutil


def write_detail_result(opt, stats_save_path, stats):
    with open(os.path.join(stats_save_path, "detail_result.txt"), "w") as f:
        metric_list = list(stats.keys())
        for i in range(len(metric_list)):
            f.write(metric_list[i] + " ")
        f.write("\n")
        for i in range(len(stats[metric_list[0]])):
            for j in range(len(metric_list)):
                f.write("{} ".format(stats[metric_list[j]][i]))
            f.write("\n")
            
            
def main(opt):
    print("preparing dataset ... ")

    test_dataset = dataloader_full(opt, mode='test')

    print('test images = {}'.format(len(test_dataset) * opt.batch_size))

    model = create_model(opt)
    
    stats_save_path = get_save_path(opt)
    os.makedirs(stats_save_path, exist_ok=True)

    print('-----------------Test Best Model-----------------')
    print("\n")
    print("load best model ...")
    model.load_networks("best")
    metrics_test, per_frame_metrics, stats = test_evaluate(opt, model, test_dataset, "best", save_result=True)
    write_detail_result(opt, stats_save_path, stats)

    print("\n")
    print("test finished !!!")
    print("\n")
    print("best test metrics:")
    overall_result_str = ""
    for k, v in metrics_test.items():
        print("{}: {:.4e}".format(k, v.item()))

    for k in metrics_test.keys():
        overall_result_str += "{} ".format(k)
    overall_result_str += "\n"
    for v in metrics_test.values():
        overall_result_str += "{} ".format(v.item())
    overall_result_str += "\n"
    shutil.copy(os.path.join(opt.log_dir, opt.experiment_name, 'test_opt.txt'),
                os.path.join('./results/', opt.experiment_name, 'test_opt.txt'))
    
    print("\n")
    
    if True:
        categorical_result_file = open(os.path.join(stats_save_path, "categorical_result.txt"), "w")
        categorical_result_file.write(overall_result_str)
        print('-----------------Start Category-Specific Evaluation-----------------')
        print("\n")
        dict_motion_category = get_dict_motion_category()
        
        for key, value in dict_motion_category.items():
            key_test_dataset = dataloader_full(opt, mode="test", id=key)
            key_metrics_test, _, _ = test_evaluate(opt, model, key_test_dataset, "best_" + key)
            print("category: {}".format(key + "_" + value))
            print("number of images: {}".format(len(key_test_dataset)))
            categorical_result_file.write("{} {} {} ".format(key, value, len(key_test_dataset)))
            for k, v in key_metrics_test.items():
                print("{}: {}".format(k, v))
                categorical_result_file.write("{} ".format(v.item()))
            categorical_result_file.write("\n")
            print("\n")
        categorical_result_file.close()

    print('-----------------All Process Finished-----------------')
    print("\n")


if __name__ == "__main__":
    opt = TestOptions().parse()
    main(opt)