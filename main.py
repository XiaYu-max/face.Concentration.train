from DataProcess import *
import argparse
import Dataset
from train import *
from visualize import *
from pytorch_model_summary import summary
def main():
    parser = argparse.ArgumentParser(prog='argparse_template.py', description='Tutorial')
    parser.add_argument('--setup', '-s', type=bool, help='第一次執行前請進行資料整理')
    parser.add_argument('--data_setup', '-ds', type=bool, help='第一次執行前請進行資料整理')

    parser.add_argument('--train', '-t', type=bool, help='訓練模型')
    parser.add_argument('--val', '-v', type=bool, help='評估模型')
    parser.add_argument('--cam', '-c', type=bool, help='啟動cam')
    parser.add_argument('--sum', '-u', type=bool, help='呈現summary')
    args = parser.parse_args()

    if args.setup:
        DataProcess()
    
    if args.data_setup:
        train_path = 'C:\\face_detection\data\\face\\train_data'
        data_label(train_path)

    train_dataset = Dataset.FaceDataset(root = 'C:\\face_detection\data\\face\\total')
    

    print('---------------dataset----------')
    if args.train:
        print('------------------training------------------')
        model = train(
            train_dataset,
            batch_size = 128,
            epochs = 100,
            learning_rate = 0.1,
            wt_decay = 0
        )
        torch.save(model.state_dict(), '.\\model\\test7.pk1')
    else :
        model = FaceCNN().to(device)
        model.load_state_dict(torch.load('.\\model\\test7.pk1'))
        model.eval()
    
    if args.val:
        acc_train = validate(model, train_dataset, 128)

    if args.cam:
        visualize(model)

    if args.sum:
        print(summary(FaceCNN(), torch.zeros((1, 1, 48, 48)), show_input=False, show_hierarchical=True))
        
if __name__ == '__main__':
    main()