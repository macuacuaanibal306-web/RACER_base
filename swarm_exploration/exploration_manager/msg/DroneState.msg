int32 drone_id

int8[] grid_ids
float64 recent_attempt_time
float64 stamp

# only used for simulation
float32[] pos
float32[] vel
float32 yaw

# predicted intent state for future rendezvous/state estimation
bool intent_valid
float32[] intent_dir
float32 intent_conf
float32 intent_sigma_rad
float32 intent_t_int
float32 intent_aligned_speed
float32 intent_aligned_speed_signed
float32 intent_speed_window
float64 intent_update_stamp
float32 intent_pred_horizon
float32[] intent_progress_vel
float32[] intent_pred_pos
