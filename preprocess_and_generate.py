"""
Pocket2Mol 预处理和生成脚本

此脚本在 Docker 容器内运行，完成以下任务：
1. 从序列化文件加载原始数据
2. 将原始数据转换为 PyG Data 格式
3. 应用 transform pipeline
4. 运行 Pocket2Mol 生成
5. 保存结果

环境要求：
- rdkit, torch_geometric, openbabel, EFGs
"""
import os
import sys
from pathlib import Path

# 添加当前目录到 Python 路径
file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)
sys.path.append(directory_path)

import logging
import argparse
import pickle
import json
import torch
import copy
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from easydict import EasyDict

from utils.molecule.constants import *
from utils.misc import seed_all
from utils.rdkit_utils import reconstruct_mol, evaluate_validity, save_mol, obabel_recover_bond
from utils.data import recursive_to
from model import Pocket2Mol

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("pocket2mol.preprocess")


# ========================================================================
# Transform Classes
# ========================================================================

def get_index(atom_num, hybridization, is_aromatic, mode):
    """获取原子类型索引"""
    if mode == 'basic':
        return map_atom_type_only_to_index[int(atom_num)]
    elif mode == 'add_aromatic':
        if (int(atom_num), bool(is_aromatic)) in map_atom_type_aromatic_to_index:
            return map_atom_type_aromatic_to_index[int(atom_num), bool(is_aromatic)]
        else:
            return map_atom_type_aromatic_to_index[(1, False)]
    else:
        return map_atom_type_full_to_index[(int(atom_num), str(hybridization), bool(is_aromatic))]


class FeaturizeProteinFullAtom:
    """蛋白质全原子特征化"""
    
    def __init__(self):
        from utils.protein.constants import atomic_numbers, aa_name_number
        self.atomic_numbers = torch.LongTensor(atomic_numbers)
        self.max_num_aa = len(aa_name_number)
    
    def __call__(self, data):
        data_prot = {}
        element = (data.protein.element.view(-1, 1) == self.atomic_numbers.view(1, -1)).float()
        amino_acid = data.protein.atom_to_aa_type
        is_backbone = data.protein.is_backbone.view(-1, 1).long()
        x = torch.cat([element, is_backbone], dim=-1)
        
        data_prot['atom_feature'] = x
        data_prot['aa_type'] = amino_acid
        data_prot['pos'] = data.protein.pos
        data_prot['element'] = data.protein.element
        data_prot['lig_flag'] = torch.zeros_like(data.protein.element, dtype=torch.bool)
        
        element_list = data.protein.element.tolist()
        data_prot['atom_type'] = torch.tensor([
            get_index(e, 0, False, 'basic') for e in element_list
        ], dtype=torch.long)
        
        data_prot['alpha_carbon_indicator'] = torch.tensor([
            name == "CA" for name in data.protein['atom_name']
        ], dtype=torch.bool)
        
        if hasattr(data.protein, 'contact'):
            data_prot['contact'] = data.protein['contact']
            data_prot['contact_idx'] = data.protein['contact_idx']
        
        data.protein = EasyDict(data_prot)
        return data


class CenterPos:
    """坐标中心化"""
    
    def __init__(self, center_flag='protein', mask_flag=None):
        self.center_flag = center_flag
        self.mask_flag = mask_flag
    
    def __call__(self, data):
        data_flag = getattr(data, self.center_flag)
        if self.mask_flag is not None and data_flag[self.mask_flag].sum() > 0:
            data_center = data_flag['pos'][data_flag[self.mask_flag]].mean(dim=0, keepdim=True)
        else:
            data_center = data_flag['pos'].mean(dim=0, keepdim=True)
        
        data.protein['pos'] = data.protein['pos'] - data_center
        data.protein['translation'] = data_center.expand(data.protein['pos'].size(0), -1)
        
        return data


class PygDatasetFromList(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ========================================================================
# 数据预处理函数
# ========================================================================

def load_raw_data(input_file):
    """加载原始数据"""
    logger.info(f"Loading raw data from {input_file}")
    with open(input_file, 'rb') as f:
        raw_data_list = pickle.load(f)
    return raw_data_list


def preprocess_protein(protein_data):
    """
    预处理蛋白质数据
    
    Args:
        protein_data: 可以是字典（包含 pdb_string）或 PyG Data 对象（包含 pos, element 等）
    """
    # 如果 protein_data 是字典且包含 pdb_string，使用 Bio.PDB 解析
    if isinstance(protein_data, dict) and 'pdb_string' in protein_data:
        from Bio.PDB import PDBParser, PDBIO
        from io import StringIO
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', StringIO(protein_data['pdb_string']))
        
        pdb_data = {
            'pos': [],
            'element': [],
            'atom_name': [],
            'atom_to_aa_type': [],
            'is_backbone': [],
        }
        
        from utils.protein.constants import aa_name_number, BBHeavyAtom
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    aa_name = residue.get_resname()
                    aa_type = aa_name_number.get(aa_name, 20)  # 20 for unknown
                    
                    for atom in residue:
                        atom_name = atom.get_name()
                        element = atom.element.upper()
                        
                        if element == '':
                            continue
                        
                        # 获取原子序数
                        from utils.protein.constants import element_to_atomic_number
                        atomic_num = element_to_atomic_number.get(element, 0)
                        
                        if atomic_num == 0:
                            continue
                        
                        pdb_data['pos'].append(atom.get_coord())
                        pdb_data['element'].append(atomic_num)
                        pdb_data['atom_name'].append(atom_name)
                        pdb_data['atom_to_aa_type'].append(aa_type)
                        pdb_data['is_backbone'].append(atom_name in BBHeavyAtom)
        
        # 转换为 tensor
        protein_tensor = EasyDict({
            'pos': torch.tensor(pdb_data['pos'], dtype=torch.float32),
            'element': torch.tensor(pdb_data['element'], dtype=torch.long),
            'atom_name': pdb_data['atom_name'],
            'atom_to_aa_type': torch.tensor(pdb_data['atom_to_aa_type'], dtype=torch.long),
            'is_backbone': torch.tensor(pdb_data['is_backbone'], dtype=torch.bool),
        })
        
        return protein_tensor
    
    # 否则，假设 protein_data 已经是 PyG Data 对象或类似结构（包括 dict 和 EasyDict）
    # 直接返回或转换为 EasyDict
    elif hasattr(protein_data, 'pos') and hasattr(protein_data, 'element'):
        # protein_data 已经是合适的格式（PyG Data 或 EasyDict）
        # 确保它有所需的所有字段
        protein_tensor = EasyDict()
        protein_tensor['pos'] = protein_data.pos if hasattr(protein_data, 'pos') else protein_data['pos']
        protein_tensor['element'] = protein_data.element if hasattr(protein_data, 'element') else protein_data['element']
        protein_tensor['atom_name'] = protein_data.atom_name if hasattr(protein_data, 'atom_name') else protein_data.get('atom_name', [])
        protein_tensor['atom_to_aa_type'] = protein_data.atom_to_aa_type if hasattr(protein_data, 'atom_to_aa_type') else protein_data.get('atom_to_aa_type', torch.zeros(len(protein_tensor['pos']), dtype=torch.long))
        protein_tensor['is_backbone'] = protein_data.is_backbone if hasattr(protein_data, 'is_backbone') else protein_data.get('is_backbone', torch.zeros(len(protein_tensor['pos']), dtype=torch.bool))
        
        return protein_tensor
    
    elif isinstance(protein_data, dict) and 'pos' in protein_data and 'element' in protein_data:
        # protein_data 是字典且包含必要字段
        protein_tensor = EasyDict()
        protein_tensor['pos'] = protein_data['pos']
        protein_tensor['element'] = protein_data['element']
        protein_tensor['atom_name'] = protein_data.get('atom_name', [])
        protein_tensor['atom_to_aa_type'] = protein_data.get('atom_to_aa_type', torch.zeros(len(protein_data['pos']), dtype=torch.long))
        protein_tensor['is_backbone'] = protein_data.get('is_backbone', torch.zeros(len(protein_data['pos']), dtype=torch.bool))
        
        return protein_tensor
    
    else:
        raise ValueError(f"Unsupported protein_data format: {type(protein_data)}. Keys: {protein_data.keys() if isinstance(protein_data, dict) else 'N/A'}")


def raw_to_pyg_data(raw_data):
    """将原始数据转换为 PyG Data 格式"""
    protein_tensor = preprocess_protein(raw_data['protein'])
    
    # 添加 batch 字段（单个样本，batch_idx=0）
    num_atoms = protein_tensor['pos'].shape[0]
    protein_tensor['batch'] = torch.zeros(num_atoms, dtype=torch.long)
    
    data = Data()
    data.protein = protein_tensor
    data.entry = raw_data['entry']
    
    return data


def apply_transforms(data):
    """应用 transform pipeline"""
    transforms = [
        FeaturizeProteinFullAtom(),
        CenterPos(center_flag='protein', mask_flag='alpha_carbon_indicator'),
    ]
    
    for transform in transforms:
        data = transform(data)
    
    return data


def translate(result, translation):
    """还原坐标平移"""
    result_pos = result[0].cpu()
    result_pos += translation.cpu()
    return [result_pos] + [result[k+1] for k in range(len(result) - 1)]


def split_batch_into_samples(batch, mode='add_aromatic'):
    """将 batch 拆分为单个样本"""
    batch_idx = batch[-1]
    if batch_idx.numel() == 0:
        return []
    B = batch_idx.max() + 1
    batch_split = []
    for i in range(B):
        idx = (batch_idx == i)
        sample = {}
        sample['pos'] = batch[0].cpu()[idx].tolist()
        sample['type'] = batch[1].cpu()[idx].numpy()
        if len(sample['type'].shape) == 2:
            sample['type'] = sample['type'].argmax(axis=-1)
        sample['atom'] = get_atomic_number_from_index(sample['type'], mode)
        sample['aromatic'] = is_aromatic_from_index(sample['type'], mode)
        batch_split.append(sample)
    return batch_split


# ========================================================================
# 主函数
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description='Pocket2Mol Preprocessing and Generation')
    
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--sampling_config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='/checkpoints/pretrained.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--threshold', type=int, default=-1)
    parser.add_argument('--threshold_ratio', type=float, default=0.8)
    
    args = parser.parse_args()
    sampling_config = json.loads(args.sampling_config)
    seed_all(sampling_config.get('seed', 2024))
    
    # 1. 加载原始数据
    raw_data_list = load_raw_data(args.input_file)
    logger.info(f"Loaded {len(raw_data_list)} samples")
    
    # 2. 预处理和转换
    logger.info("Preprocessing data...")
    processed_data = []
    for raw_data in tqdm(raw_data_list, desc="Preprocessing"):
        try:
            # 转换为 PyG Data
            data = raw_to_pyg_data(raw_data)
            # 应用 transforms
            data = apply_transforms(data)
            processed_data.append({
                'trans_data': data,
                'entry': raw_data['entry']
            })
        except Exception as e:
            logger.error(f"Error processing {raw_data['entry']}: {e}")
            continue
    
    logger.info(f"Successfully preprocessed {len(processed_data)} samples")
    
    # 3. 加载模型
    logger.info(f'Loading model from: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = Pocket2Mol(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    logger.info(str(lsd))
    
    # 4. 生成分子
    logger.info("Starting molecule generation...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    for sample in processed_data:
        structure_id = sample['entry'][0]
        if structure_id.endswith('.pdb'):
            structure_id = structure_id[:-4]
        
        save_dir = os.path.join(args.output_dir, structure_id)
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f'Generating for: {structure_id}')
        
        # 复制样本以生成多个候选
        sample_list_repeat = [
            copy.deepcopy(sample['trans_data']) 
            for _ in range(sampling_config['num_samples'])
        ]
        
        batch_size = sampling_config.get('batch_size', args.batch_size)
        loader = DataLoader(
            PygDatasetFromList(sample_list_repeat),
            batch_size=batch_size,
            shuffle=False
        )
        
        count = 0
        for batch in tqdm(loader, desc=structure_id, dynamic_ncols=True):
            torch.set_grad_enabled(False)
            model.eval()
            
            # 移动到设备
            # 注意：batch 是 DataBatch 对象，recursive_to 不会处理它
            # 需要手动移动 batch.protein 中的所有 tensor
            if hasattr(batch, 'protein') and isinstance(batch.protein, dict):
                for key, value in batch.protein.items():
                    if isinstance(value, torch.Tensor):
                        batch.protein[key] = value.to(args.device)
            
            # 手动为 batch.protein 添加 batch 索引
            # PyG DataLoader 不会自动为嵌套的 dict 添加 batch 索引
            if hasattr(batch, 'protein') and isinstance(batch.protein, dict):
                total_atoms = batch.protein['pos'].shape[0]
                atoms_per_sample = total_atoms // len(batch.entry)  # 使用 entry 数量作为 batch size
                
                batch_indices = []
                for i in range(len(batch.entry)):
                    batch_indices.extend([i] * atoms_per_sample)
                
                batch.protein['batch'] = torch.tensor(batch_indices, dtype=torch.long, device=args.device)
            
            # 创建空的 ligand_context（从头生成分子）
            batch.ligand_context = EasyDict({
                'atom_type': torch.empty(0, dtype=torch.long, device=args.device),
                'pos': torch.empty((0, 3), dtype=torch.float32, device=args.device),
                'batch': torch.empty(0, dtype=torch.long, device=args.device),
                'lig_flag': torch.empty(0, dtype=torch.bool, device=args.device),
            })
            
            # 添加空的边信息
            batch[('ligand_context', 'to', 'ligand_context')] = EasyDict({
                'bond_index': torch.empty((2, 0), dtype=torch.long, device=args.device),
                'bond_type': torch.empty(0, dtype=torch.long, device=args.device),
            })
            
            traj_batch = model.sample(batch)
            if len(traj_batch) == 0:
                continue
            
            if sampling_config.get('translate', True):
                # translation 字段在 batch.protein 字典中
                result_batch = translate(traj_batch[0], batch.protein['translation'][:1])
            else:
                result_batch = traj_batch[0]
            
            result_split = split_batch_into_samples(result_batch, mode=sampling_config.get('mode', 'add_aromatic'))
            
            if sampling_config.get('reconstruct', None) is not None:
                for result in result_split:
                    try:
                        try:
                            mol = reconstruct_mol(
                                result['pos'],
                                result['atom'],
                                result['aromatic'],
                                basic_mode=sampling_config['reconstruct'].get('basic_mode', 'ref_angles')
                            )
                        except:
                            mol = obabel_recover_bond(result['pos'], result['atom'])
                        
                        mol, success = evaluate_validity(mol, args.threshold, args.threshold_ratio)
                        if success:
                            count += 1
                            data = {
                                'pos': np.array(result['pos']),
                                'atom': np.array(result['atom']),
                                'entry': sample['entry']
                            }
                            torch.save(data, os.path.join(save_dir, f'sample_{count:04d}.pt'))
                            save_mol(mol, os.path.join(save_dir, f'sample_{count:04d}.sdf'))
                    except Exception as e:
                        logger.debug(f"Failed to reconstruct molecule: {e}")
                        continue
        
        logger.info(f"Generated {count} valid molecules for {structure_id}")
    
    logger.info("Generation complete!")


if __name__ == '__main__':
    main()
