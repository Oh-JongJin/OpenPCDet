import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        
    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)

            # print(self.sample_file_list[index])
            # points = np.fromfile(self.sample_file_list[index], dtype=np.float32)
            # if points.size % 4 != 0:
                # data = (points.size // 4) * 4
                # points = points[:data]
            # points = points.reshape(-1, 4)

        elif self.ext == '.npy':
            data = np.load(self.sample_file_list[index])
            points = data.view(np.float32).reshape(-1, 4)
            # data = data[:192]
            # # points = data.reshape(-1, 4)
            # points = np.reshape(data, [-1, 4])

            # # Test array create
            # lst = []
            # for i in range(0, 1500):
            #     lst.append(i / 1500)
            # data = np.array(lst, dtype=np.float32)
            # # data = np.array([10,896.0, -1.569, -2.449], dtype=np.float32)
            # points = np.reshape(data, [-1, 4])

        # elif self.ext == '.pcd':
        #     pcd = open3d.io.read_point_cloud(self.sample_file_list[index])
        #     # pcd_points = np.asarray(pcd.points, dtype=np.float32)
        #     pcd_to_bin = []
        #     for i in points:
        #         values = i.tolist()
        #         values.append(0.)
        #         pcd_to_bin.append(values)
        #     pcd_to_bin = self.flatten(pcd_to_bin)
        #     points = np.array(pcd_to_bin, dtype=np.float32)

        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
    def flatten(self, lst):
        result = []
        for item in lst:
            result.extend(item)
        return result


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            # pred_dicts, _ = model.forward(data_dict)

            # Draw predict rectangle
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            V.draw_scenes(
                points=data_dict['points'][:, 1:]
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
