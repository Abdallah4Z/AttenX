import torch
import os
import re

def parse_logs():
    variants = ['baseline', 'attenx_sa', 'attenx_mhsa']
    base_path = 'results/fair_parallel_50e_cub'
    
    print('| Variant | Ckpt Epoch | Latest Step | D_Loss | G_Loss | L_KL | Gamma_DAMSM | Latest Epoch Marker | Recent Checkpoints |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    
    for var in variants:
        var_path = os.path.join(base_path, var)
        
        # 1. Epoch from checkpoint_latest.pth
        ckpt_path = os.path.join(var_path, 'checkpoints', 'checkpoint_latest.pth')
        epoch_val = 'N/A'
        if os.path.exists(ckpt_path):
            try:
                # Use CPU mapping to avoid GPU issues
                ckpt = torch.load(ckpt_path, map_location='cpu')
                epoch_val = ckpt.get('epoch', 'N/A')
            except Exception as e:
                epoch_val = 'Err'
        
        # 2 & 3. Parse train.out
        train_out = os.path.join(var_path, 'train.out')
        latest_step, d_loss, g_loss, l_kl, gamma, latest_marker = ['N/A']*6
        if os.path.exists(train_out):
            with open(train_out, 'r') as f:
                for line in reversed(f.readlines()):
                    if 'Step [' in line and latest_step == 'N/A':
                        s_m = re.search(r'Step\s+\[(\d+)\]', line)
                        d_m = re.search(r'D_Loss:\s+([\d\.]+)', line)
                        g_m = re.search(r'G_Loss:\s+([\d\.]+)', line)
                        k_m = re.search(r'L_KL:\s+([\d\.]+)', line)
                        ga_m = re.search(r'Gamma_DAMSM:\s+([\d\.]+)', line)
                        if s_m: latest_step = s_m.group(1)
                        if d_m: d_loss = d_m.group(1)
                        if g_m: g_loss = g_m.group(1)
                        if k_m: l_kl = k_m.group(1)
                        if ga_m: gamma = ga_m.group(1)
                    
                    if 'complete' in line.lower() and 'Epoch' in line and latest_marker == 'N/A':
                        m = re.search(r'Epoch\s+\d+/50\s+complete', line)
                        if m: latest_marker = m.group(0)

        # 4. Recent checkpoints
        ckpt_dir = os.path.join(var_path, 'checkpoints')
        recent = 'N/A'
        if os.path.exists(ckpt_dir):
            files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
            recent = '<br>'.join(files[:3])
            
        print(f'| {var} | {epoch_val} | {latest_step} | {d_loss} | {g_loss} | {l_kl} | {gamma} | {latest_marker} | {recent} |')

if __name__ == "__main__":
    parse_logs()
