import time
import argparse
import torch

from tqdm import tqdm
from pathlib import Path

from eval_utils import load_model


def get_embedding(loader, model):
    matids = []
    z_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx} in {len(loader)}')
        # collect material ID and embedding vector
        matids += batch['matid']
        _, _, z = model.encode(batch)
        z_list.append(z.detach())
    zs = torch.cat(z_list, 0)
    return matids, zs


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(model_path, load_data=True)

    if torch.cuda.is_available():
        model.to('cuda')

    print('Compute encoder embedding for structures.')
    start_time = time.time()

    matids, zs = get_embedding(test_loader, model)

    if args.label == '':
        embed_out_name = 'embed.pt'
    else:
        embed_out_name = f'embed_{args.label}.pt'

    torch.save({
        'mat_id': matids,
        'embedding': zs,
        'time': time.time() - start_time
    }, model_path / embed_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--label', default='')

    args = parser.parse_args()

    main(args)
