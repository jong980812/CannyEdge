def arg_print(args):
    print("-------------------------------Argument------------------------------------")
    for k, v in vars(args).items():
        print(f'{k}={v}')
    print("---------------------------------------------------------------------------")
