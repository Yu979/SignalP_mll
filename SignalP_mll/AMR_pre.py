import argparse


if __name__ == '__main__':
    # 参数定义
    parser = argparse.ArgumentParser(description='PyTorch AMR Model')
    parser.add_argument('--model_arch', default="CNN", type=str, help='the architecture of model')
    parser.add_argument('--seed', default="None", type=int, help='seed for initializing training.')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--path', default=None, type=str,
                        help='File path')

    print("processing...")
    print("================================================================")
    print("Cefoxitin - Susceptible"
          '\n'
          "cefotaxime - Resistant"
          '\n'
          "tetracycline - Resistant"
          '\n'
          "ertapenem - Susceptible"
          '\n'
          "cefotaxime/clavulanic acids - Resistant"
          '\n'
          "gentamicin - Resistant"
          '\n'
          "ciprofloxacin - Resistant")
    print("================================================================")
