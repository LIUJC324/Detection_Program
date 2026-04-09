# DroneVehicle 数据与标签交接说明

## 1. 当前数据集位置

已下载和解压的位置：

- 原始压缩包目录：`/home/liujuncheng/rgbt_uav_detection/datasets/raw/dronevehicle`
- 当前已解压验证目录：`/home/liujuncheng/rgbt_uav_detection/datasets/raw/dronevehicle/val_unpack/val`

其中：

- `valimg/`：RGB 图像
- `valimgr/`：Thermal 图像
- `vallabel/`：RGB 标注 XML
- `vallabelr/`：Thermal 标注 XML

## 2. 当前标签定义

当前项目已切换为 DroneVehicle 5 类：

```json
{
  "0": "car",
  "1": "truck",
  "2": "bus",
  "3": "van",
  "4": "freight_car"
}
```

说明：

- 原始数据里还会出现 `feright car` / `feright_car`，项目内统一标准化为 `freight_car`

## 3. 给前后端展示的样例怎么导出

执行：

```bash
cd /home/liujuncheng/rgbt_uav_detection
python3 scripts/export_dronevehicle_showcase.py --sample-id 00001
```

默认会输出到：

- `/home/liujuncheng/rgbt_uav_detection/outputs/showcase/dronevehicle_00001`

导出内容包括：

- `rgb.jpg`
- `thermal.jpg`
- `rgb_overlay.jpg`
- `thermal_overlay.jpg`
- `rgb_annotations.json`
- `thermal_annotations.json`
- `rgb_raw.xml`
- `thermal_raw.xml`
- `showcase_manifest.json`

推荐给前后端看的文件：

- `rgb_overlay.jpg`
- `thermal_overlay.jpg`
- `showcase_manifest.json`
- `rgb_annotations.json`
- `thermal_annotations.json`

## 4. 是否需要通知前后端修改接口

结论：

- `接口字段本身不用改`
- `标签字典和前端展示文案需要改`

原因：

- `POST /v1/detect/stream` 的返回结构没有变，仍然是 `bbox / confidence / class_id / class_name`
- `GET /v1/model/info` 的返回结构也没有变，仍然提供 `class_mapping`
- 变化的是 `class_id` 对应的语义，从旧的 `person / tricycle` 改成了 `van / freight_car`

因此需要通知前后端同步的内容是：

- 不要再写死旧标签
- 前端图例、筛选项、颜色映射要改
- 后端如果有枚举、数据库字典、统计报表、告警规则，也要改

如果前后端已经按 `GET /v1/model/info.class_mapping` 动态读取标签，则不需要改接口代码，只需要重新联调确认展示即可。
