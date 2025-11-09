import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from main import load_data, load_model, N_STEPS, DEVICE

def get_predictions(rollout_steps=1000):
    test_dataset, test_data_scaled, scaler, n_features = load_data()
    model = load_model(in_channels=n_features, out_channels=n_features)
    print(f"Generating predictions for the first sequence with {rollout_steps} steps...")
    initial_sequence = test_dataset[0]['x'].unsqueeze(0).to(DEVICE)
    predictions = []
    current_sequence = initial_sequence.clone()
    with torch.no_grad():
        for _ in range(rollout_steps):
            pred = model(current_sequence)
            next_step_pred = pred[:, :, 0:1]
            predictions.append(next_step_pred.cpu().numpy().squeeze())
            current_sequence = torch.cat([current_sequence[:, :, 1:], next_step_pred], dim=2)
    predictions = np.array(predictions)
    predictions_rescaled = scaler.inverse_transform(predictions)
    ground_truth_rescaled = scaler.inverse_transform(test_data_scaled[:rollout_steps])
    return ground_truth_rescaled, predictions_rescaled

def create_animation(actual, predictions):
    print("Creating animation...")
    body1_actual = actual[:, 0:3]
    body2_actual = actual[:, 3:6]
    body3_actual = actual[:, 6:9]
    body1_pred = predictions[:, 0:3]
    body2_pred = predictions[:, 3:6]
    body3_pred = predictions[:, 6:9]
    fig = plt.figure(figsize=(16, 8))
    ax_left = fig.add_subplot(121, projection='3d')
    ax_right = fig.add_subplot(122, projection='3d')
    n_frames = actual.shape[0]
    def update(frame):
        ax_left.clear()
        ax_right.clear()
        l1, = ax_left.plot(body1_actual[:frame, 0], body1_actual[:frame, 1], body1_actual[:frame, 2], color='blue', label='Body 1 Actual')
        l2, = ax_left.plot(body2_actual[:frame, 0], body2_actual[:frame, 1], body2_actual[:frame, 2], color='green', label='Body 2 Actual')
        l3, = ax_left.plot(body3_actual[:frame, 0], body3_actual[:frame, 1], body3_actual[:frame, 2], color='red', label='Body 3 Actual')
        ax_left.set_xlabel('X')
        ax_left.set_ylabel('Y')
        ax_left.set_zlabel('Z')
        ax_left.set_title('Actual Bodies')
        ax_left.legend(loc='upper right')
        ax_left.set_xlim(np.min(actual[:, [0, 3, 6]]), np.max(actual[:, [0, 3, 6]]))
        ax_left.set_ylim(np.min(actual[:, [1, 4, 7]]), np.max(actual[:, [1, 4, 7]]))
        ax_left.set_zlim(np.min(actual[:, [2, 5, 8]]), np.max(actual[:, [2, 5, 8]]))
        l4, = ax_right.plot(body1_pred[:frame, 0], body1_pred[:frame, 1], body1_pred[:frame, 2], color='orange', label='Body 1 Predicted')
        l5, = ax_right.plot(body2_pred[:frame, 0], body2_pred[:frame, 1], body2_pred[:frame, 2], color='purple', label='Body 2 Predicted')
        l6, = ax_right.plot(body3_pred[:frame, 0], body3_pred[:frame, 1], body3_pred[:frame, 2], color='brown', label='Body 3 Predicted')
        ax_right.set_xlabel('X')
        ax_right.set_ylabel('Y')
        ax_right.set_zlabel('Z')
        ax_right.set_title('Predicted Bodies')
        ax_right.legend(loc='upper right')
        ax_right.set_xlim(np.min(actual[:, [0, 3, 6]]), np.max(actual[:, [0, 3, 6]]))
        ax_right.set_ylim(np.min(actual[:, [1, 4, 7]]), np.max(actual[:, [1, 4, 7]]))
        ax_right.set_zlim(np.min(actual[:, [2, 5, 8]]), np.max(actual[:, [2, 5, 8]]))
        return l1, l2, l3, l4, l5, l6
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=10, blit=True)
    save_path = 'Three_Body_Problem/FNO/three_body_simulation_split.mp4'
    ani.save(save_path, writer='ffmpeg')
    print(f"Video saved as {save_path}")

if __name__ == '__main__':
    actual_data, predicted_data = get_predictions(rollout_steps=1000)
    if actual_data is not None and predicted_data is not None:
        create_animation(actual_data, predicted_data)
    else:
        print("Could not generate predictions. Please check the data and model.")
