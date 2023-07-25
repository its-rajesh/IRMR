import argparse
import numpy as np
import os
import librosa as lb
from tqdm import tqdm
import shutil


class artificial_mixture:

    def random_gen(self):
        k = np.round(np.random.uniform(0.05, 0.31), 3)
        return k
    
    def random_diag(self):
        k = np.round(np.random.uniform(0.6, 1), 3)
        return k

    def gen_A(self):
        A = np.array([[self.random_diag(), self.random_gen(), self.random_gen(), self.random_gen()],
                    [self.random_gen(), self.random_diag(), self.random_gen(), self.random_gen()],
                    [self.random_gen(), self.random_gen(), self.random_diag(), self.random_gen()],
                    [self.random_gen(), self.random_gen(), self.random_gen(), self.random_diag()]])
        return np.round(A, 2)
    
    def conditions(self, vx, bx, dx, block):
        flag = True
        if vx.shape[0] != block:
            flag = False
        if np.linalg.norm(vx) <= 0.9 or np.linalg.norm(bx) <= 0.9 or np.linalg.norm(dx) <= 0.9:
            flag = False
        return flag
    

    def delete_folder(self, folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' and its contents have been deleted.")
        except FileNotFoundError:
            print(f"Folder '{folder_path}' not found.")
        except Exception as e:
            print(f"An error occurred while deleting the folder: {e}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MUSDB18HQ Artificial Mixing")

    # Add dataset path
    parser.add_argument("-dataset", "--path", type=str, help="Raw MUSDB18HQ dataset")
    args = parser.parse_args()

    files = sorted(os.listdir(os.path.join(args.path, 'train')))
    am = artificial_mixture()

    for file in tqdm(files):
        vocal, fs = lb.load(args.path+file+"/vocals.wav")
        bass, fs = lb.load(args.path+file+"/bass.wav")
        drums, fs = lb.load(args.path+file+"/drums.wav")
        other, fs = lb.load(args.path+file+"/other.wav")
        
        path = './temp/'+file
        os.makedirs(path, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        A = am.gen_A()
        np.save(path+'/mixing.npy', A)
        
        sec = 10
        block = sec*fs
        length = vocal.shape[0]
        for i in range(0, length, block):
            vx = vocal[i:i+block]
            bx = bass[i:i+block]
            dx = drums[i:i+block]
            ox = other[i:i+block]

            if am.conditions(vx, bx, dx, block):

                fnpath = path+'/{}_{}/'.format(int(i/fs), int(i/fs)+sec)
                os.makedirs(fnpath, exist_ok=True)

                np.save(fnpath+'vocal.npy', vx)
                np.save(fnpath+'bass.npy', bx)
                np.save(fnpath+'drums.npy', dx)
                np.save(fnpath+'other.npy', ox)

                s = np.array([vx, bx, dx, ox])
                x = A @ s

                np.save(fnpath+'bvocal.npy', x[0])
                np.save(fnpath+'bbass.npy', x[1])
                np.save(fnpath+'bdrums.npy', x[2])
                np.save(fnpath+'bother.npy', x[3])

    
    print('Merging numpys from temp')
    mainfolder = sorted(os.listdir(path))
    xtrain, ytrain = [], []
    for folder in tqdm(mainfolder[30:50]):
        dept_path = path+folder+'/'
        subfolder = sorted(os.listdir(dept_path))
        for files in subfolder:
            v = np.load(dept_path+files+'/vocal.npy')
            b = np.load(dept_path+files+'/bass.npy')
            d = np.load(dept_path+files+'/drums.npy')
            o = np.load(dept_path+files+'/other.npy')
            
            bv = np.load(dept_path+files+'/bvocal.npy')
            bb = np.load(dept_path+files+'/bbass.npy')
            bd = np.load(dept_path+files+'/bdrums.npy')
            bo = np.load(dept_path+files+'/bother.npy')
            
            S = np.array([v, b, d, o])
            X = np.array([bv, bb, bd, bo])
            xtrain.append(X)
            ytrain.append(S)
            
    out = "./numpy_files/"
    np.save(out+'Xtrain.npy', xtrain)
    np.save(out+'Ytrain.npy', ytrain)
    print('temp files deleted, data generated')
    print('Check ./numpy_files folder for further processing')
    am.delete_folder('./temp/')

    



