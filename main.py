# ... (вверху файла, после других импортов) ...

# ---------- Новые API для управления камерой и трекингом ----------
@app.route('/api/camera/params', methods=['GET', 'POST'])
def camera_params():
    """Получить или установить параметры камеры (depth_enabled, tracking и т.д.)"""
    if camera is None:
        return jsonify({'error': 'Camera not initialized'}), 500

    if request.method == 'GET':
        with camera.lock:
            params = {
                'depth_enabled': camera.depth_enabled,
                'face_tracking_enabled': camera.face_tracking_enabled,
                'tracking_scale_x': camera.tracking_scale_x,
                'tracking_scale_y': camera.tracking_scale_y,
                'tracking_offset_x': camera.tracking_offset_x,
                'tracking_offset_y': camera.tracking_offset_y,
                'alpha_depth': camera.alpha_depth,
                'show_left': camera.show_left,
                'num_disp': camera.num_disp,
            }
        return jsonify(params)
    else:  # POST
        data = request.json
        camera.update_params(
            depth_enabled=data.get('depth_enabled'),
            face_tracking_enabled=data.get('face_tracking_enabled'),
            tracking_scale_x=data.get('tracking_scale_x'),
            tracking_scale_y=data.get('tracking_scale_y'),
            tracking_offset_x=data.get('tracking_offset_x'),
            tracking_offset_y=data.get('tracking_offset_y'),
            alpha_depth=data.get('alpha_depth'),
            show_left=data.get('show_left'),
            num_disp=data.get('num_disp')
        )
        return jsonify({'status': 'ok'})

@app.route('/api/tracking/offsets')
def tracking_offsets():
    """Возвращает текущие смещения для глаз (dx, dy) в пикселях."""
    if camera is None:
        return jsonify({'dx': 0, 'dy': 0})
    dx, dy = camera.get_eye_offsets()
    return jsonify({'dx': dx, 'dy': dy})

# ---------- Остальные маршруты без изменений ----------
# ... (все остальные функции остаются как есть) ...
