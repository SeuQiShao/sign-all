import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from model.modules import *
from model.GSIDecoder import DGSIDecoder, CGSIDecoder
from model.BasisClassifier import GCN_Classifier, MLP_Classifier
from model import utils


def load_decoder(args):
    if args.decoder == "DGSI":
        decoder = DGSIDecoder(
            args,
    )
    elif args.decoder == "CGSI":
        decoder = CGSIDecoder(
            args,
    )

    decoder, num_GPU = utils.distribute_over_GPUs(args, decoder, num_GPU=args.num_GPU)


    if args.load_folder:
        print("Loading model file")
        args.decoder_file = os.path.join(args.load_folder, "decoder.pt")
        decoder.load_state_dict(torch.load(args.decoder_file, map_location=args.device))
        args.save_folder = False

    return decoder


def load_model(args):

    decoder = load_decoder(args)


    #     )

    #         optimizer,


    #        encoder, decoder, optimizer, scheduler
    #     )

    encoder = None

    optimizer = optim.Adam(
        list(decoder.parameters()),
        lr=args.lr,
    )
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay, gamma=args.gamma
    )
    return (
        encoder, decoder, optimizer, scheduler
    )


if __name__ == '__main__':
    pass

