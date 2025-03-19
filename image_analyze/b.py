import argparse
import matplotlib
import fence
import gamma
import overlap
import rec
import star

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析类型')
    parser.add_argument('--type', type=str, help='[rec, star, fence, overlap, gamma]',
                        choices=['rec', 'star', 'fence', 'overlap', "gamma"], required=True)

    args = parser.parse_args()
    args_type = args.type

    if args_type == 'rec':
        rec.main()
    elif args_type == 'star':
        star.main()
    elif args_type == 'fence':
        fence.main()
    elif args_type == 'overlap':
        overlap.main()
    elif args_type == 'gamma':
        gamma.main()
