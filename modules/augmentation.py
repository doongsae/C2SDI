############################################################
#                      augmentation.py                     #
#        Make augmentation by polynomial regression        #
############################################################

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)



#### 증강 방법 검증용 함수
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

def visualize_augmented_3d(original_data, augmented_data, num_aug, save_dir='visualization_results'):
    """
    원본 데이터와 증강된 데이터를 3D로 시각화하고 저장합니다.
    
    Args:
        original_data: 원본 데이터 (첫 번째 데이터만 사용)
        augmented_data: 증강된 전체 데이터
        num_aug: 각 데이터당 증강된 개수
        save_dir: 이미지를 저장할 디렉토리
    """
    # 저장 디렉토리 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 현재 시간을 파일명에 포함
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 첫 번째 원본 데이터와 그에 해당하는 증강 데이터만 선택
    orig = original_data[0]
    augs = augmented_data[1:num_aug+1]
    
    # 첫 번째 시점에서의 시각화
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 원본 데이터 플롯 (빨간색, 더 굵게)
    mask = ~np.isnan(orig).any(axis=1)
    ax.plot(orig[mask, 0], orig[mask, 1], orig[mask, 2], 
            'r-', linewidth=3, label='Original')
    
    # 증강된 데이터 플롯 (각각 다른 색상)
    colors = plt.cm.viridis(np.linspace(0, 1, num_aug))
    for i, aug_data in enumerate(augs):
        mask = ~np.isnan(aug_data).any(axis=1)
        ax.plot(aug_data[mask, 0], aug_data[mask, 1], aug_data[mask, 2],
                '-', color=colors[i], linewidth=1, 
                label=f'Augmented {i+1}', alpha=0.7)
    
    # 그래프 꾸미기
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Original vs Augmented Trajectories')
    
    # 범례 위치 조정
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    # 보기 각도 설정
    ax.view_init(elev=20, azim=45)
    
    # 첫 번째 시점 이미지 저장
    plt.tight_layout()
    save_path_1 = os.path.join(save_dir, f'augmentation_view1_{timestamp}.png')
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close()

    # 두 번째 시점에서의 시각화
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 원본 데이터 플롯
    mask = ~np.isnan(orig).any(axis=1)
    ax.plot(orig[mask, 0], orig[mask, 1], orig[mask, 2], 
            'r-', linewidth=3, label='Original')
    
    # 증강된 데이터 플롯
    for i, aug_data in enumerate(augs):
        mask = ~np.isnan(aug_data).any(axis=1)
        ax.plot(aug_data[mask, 0], aug_data[mask, 1], aug_data[mask, 2],
                '-', color=colors[i], linewidth=1, 
                label=f'Augmented {i+1}', alpha=0.7)
    
    # 그래프 꾸미기
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Original vs Augmented Trajectories (Different View)')
    
    # 범례 위치 조정
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    # 다른 각도에서 보기
    ax.view_init(elev=0, azim=90)
    
    # 두 번째 시점 이미지 저장
    plt.tight_layout()
    save_path_2 = os.path.join(save_dir, f'augmentation_view2_{timestamp}.png')
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"이미지가 저장되었습니다:")
    print(f"View 1: {save_path_1}")
    print(f"View 2: {save_path_2}")



def fit_polynomial(t, values, degree):
    # Choose non-NaN values
    mask = ~np.isnan(values)
    t_valid = t[mask]
    values_valid = values[mask]
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(t_valid.reshape(-1, 1), values_valid)
    return model

def augment_polyreg(data, num_augmentations, scale=True):
    seq_length, num_features = data.shape
    t = np.linspace(0, 1, seq_length)
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]
    augmented_datasets = []

    # 더 높은 차수의 다항식 사용
    # polynomial_degree = 12  # 8에서 12로 증가
    polynomial_degree = 8     # 다시 감소
    
    # Fitting
    model_x = fit_polynomial(t, X, polynomial_degree)
    model_y = fit_polynomial(t, Y, polynomial_degree)
    model_z = fit_polynomial(t, Z, polynomial_degree)

    for _ in range(num_augmentations):
        augmented_data = data.copy()
        
        # 노이즈 범위를 더 크게 설정 (-2, 2로 증가)
        adjust_x = np.random.uniform(-2, 2, polynomial_degree + 1)
        adjust_y = np.random.uniform(-2, 2, polynomial_degree + 1)
        adjust_z = np.random.uniform(-2, 2, polynomial_degree + 1)

        # 추가적인 랜덤 스케일링 팩터 적용
        if scale:
            scale_factor = np.random.uniform(0.8, 1.2)
        else:
            scale_factor = 1
        
        # 랜덤 시프트 추가
        shift_x = np.random.uniform(-0.5, 0.5) * np.std(X[~np.isnan(X)])
        shift_y = np.random.uniform(-0.5, 0.5) * np.std(Y[~np.isnan(Y)])
        shift_z = np.random.uniform(-0.5, 0.5) * np.std(Z[~np.isnan(Z)])

        # predict for non-NaN values
        mask_x = ~np.isnan(X)
        mask_y = ~np.isnan(Y)
        mask_z = ~np.isnan(Z)

        x_pred = np.full_like(X, np.nan)
        y_pred = np.full_like(Y, np.nan)
        z_pred = np.full_like(Z, np.nan)

        x_pred[mask_x] = model_x.predict(t[mask_x].reshape(-1, 1)).flatten()
        y_pred[mask_y] = model_y.predict(t[mask_y].reshape(-1, 1)).flatten()
        z_pred[mask_z] = model_z.predict(t[mask_z].reshape(-1, 1)).flatten()
        
        # 스케일링과 시프트를 포함한 증강
        augmented_data[:, 0] = np.where(mask_x, 
            scale_factor * (x_pred + np.polyval(adjust_x[::-1], t)) + shift_x, X)
        augmented_data[:, 1] = np.where(mask_y, 
            scale_factor * (y_pred + np.polyval(adjust_y[::-1], t)) + shift_y, Y)
        augmented_data[:, 2] = np.where(mask_z, 
            scale_factor * (z_pred + np.polyval(adjust_z[::-1], t)) + shift_z, Z)

        augmented_datasets.append(augmented_data)

    return np.array(augmented_datasets)

def augmentation(base_data, num_aug, scale=True):
    augmented = []
    logger.info(f"Augmenting {num_aug} samples for each datum ...")
    
    # 원본 데이터 저장
    original_data = np.array(base_data)
    
    # 데이터 증강 수행
    for datum in base_data:
        augmented.append(datum)
        augmented.extend(augment_polyreg(datum, num_augmentations=num_aug, scale=scale))
    augmented_data = np.array(augmented)
    
    # 디버깅 정보 출력
    logger.info("\n=== 데이터 증강 디버깅 정보 ===")
    logger.info(f"원본 데이터 개수: {len(original_data)}")
    logger.info(f"증강된 데이터 개수: {len(augmented_data)}")
    
    # 데이터 포인트 샘플 비교
    logger.info("\n[데이터 포인트 샘플 비교]")
    n_samples = 3
    sample_timesteps = [0, len(original_data[0])//8, len(original_data[0])//6, 
                        len(original_data[0])//4, len(original_data[0])//2, 
                        len(original_data[0])//8 + len(original_data[0])//2, len(original_data[0])//6 + len(original_data[0])//2,
                        len(original_data[0])//4 +len(original_data[0])//2, -1]  # 시작, 중간, 끝
    
    for i in range(min(n_samples, len(original_data))):
        logger.info(f"\n샘플 {i+1} 비교:")
        orig_sample = original_data[i]
        aug_sample1 = augmented_data[i*(num_aug+1)+1]  # 첫 번째 증강 데이터
        aug_sample2 = augmented_data[i*(num_aug+1)+2]  # 두 번째 증강 데이터
        aug_sample3 = augmented_data[i*(num_aug+1)+3]  # 세 번째 증강 데이터
        for t in sample_timesteps:
            logger.info(f"타임스텝 {t}:")
            logger.info(f"  원본: {orig_sample[t]}")
            logger.info(f"  증강1: {aug_sample1[t]}")
            logger.info(f"  증강2: {aug_sample2[t]}")
            logger.info(f"  증강3: {aug_sample3[t]}")
            
    
    # augmentation 함수 실행 후
    # visualize_augmented_3d(original_data, augmented_data, num_aug=num_aug)
    
    return augmented_data