# AT Inspector · 空三诊断工作台

独立于算法仓库的本机/内网查看器。代码、构建文件和缓存均位于 `/root/at_unconverance/script/frontend`。只读取输入项目和 `/code` 中现有的 OpenSfM 原生模块，不调用算法启动脚本，不修改、清理或重跑项目。

## 启动

已为当前机器创建独立 Conda 环境 `at_viewer`，Python 3.9 与仓库的 OpenSfM 扩展 ABI 一致。

```bash
cd /root/at_unconverance/script/frontend
bash run_viewer.sh /root/at_unconverance/example_data1 --host 0.0.0.0 --port 8080
```

在浏览器访问 `http://服务器IP:8080`。容器环境需要将该端口映射到宿主机；`127.0.0.1` 指运行浏览器的那台机器。若通过 SSH 使用，可保留默认监听地址并建立端口转发：

```bash
# 服务器上
bash run_viewer.sh /root/at_unconverance/example_data1
# 电脑上
ssh -L 8080:127.0.0.1:8080 用户名@服务器IP
```

随后电脑访问 `http://127.0.0.1:8080`。Ctrl+C 停止服务。默认监听 `127.0.0.1:8080`，面向可信内网，不提供公网账号权限系统。

启动参数：

- `--host`、`--port`：监听地址和端口。
- `--cache-dir`：默认 `<查看器目录>/.cache`，与数据项目分离。
- `--reconstruction /绝对路径/重建文件.json`：明确选择重建文件。
- `--reindex`：重建缓存。网页右上角刷新按钮也可触发。
- `AT_VIEWER_PYTHON`：覆盖 Conda 环境 Python 路径，默认 `/root/miniconda3/envs/at_viewer/bin/python`。
- `AT_VIEWER_ODM_ROOT`：算法安装根目录，默认 `/code`，仅用于只读加载 OpenSfM。

## 界面使用（交互优化版）

1. **三维重建**：默认显示所有相机的完整视锥，“全部视锥”滑块统一调整其大小，不随距离自动简化。普通相机呈鲜绿色，选中或悬停相机呈亮青色，与它直接匹配的相机呈黄色高亮，不绘制三维匹配连线。悬停显示缩略图；当鼠标约 10 像素范围内有多个相机时，提供候选列表，这表示屏幕位置接近，不表示三维位置相同。支持选择相机/三维点模式、相机范围筛选、定位和误差着色。
2. **图像特征**：侧栏点击“展开查看”进入大图。滚轮围绕鼠标缩放，左键拖动，支持适应窗口、原图 100% 和浏览器全屏；100% 表示一原图像素对应一 CSS 像素。点击特征仅就地显示详情，主动点击“查看 Track”或“定位三维点”才跳转。大图可调点大小、透明度、按状态着色；选中点可以显示预测投影线。
3. **双图匹配**：默认展示 200 条固定抽样连线；可选择 100/200/500/全部或仅选中匹配。可拖动四个滑块分别设置连线不透明度/线宽、匹配点不透明度/直径；默认分别为 80%、1.2 px、100%、6 px。0% 表示完全透明，100% 表示不透明；当前页面会话内关闭重开窗口仍保留设置，可一键恢复默认值。左右图独立平移缩放，也可联动缩放；点击端点后可定位两个端点，或在内嵌面板查看 Track。返回后保持两幅图的观察位置。
4. **Tracks 变化**：比较匹配与最终 Tracks 的长度迁移、逐图保留率趋势、图像对连接变化及 Track 去向。支持跳转图像、双图对照和具体观测。当前分量模式固定相同的已注册图像范围；全项目模式包含未注册图像。
5. **重建点误差诊断**：每行对应一个重建点。默认以每点平均观测误差排序、筛选和着色。点击记录定位三维点，再点击观测缩略图打开原分辨率图像并定位特征；圆圈为观测，十字为预测投影。
6. **布局与状态**：顶部可折叠两侧面板，拖动分隔条调整宽度。切换页面保留三维视角；关闭、重新展开图像或匹配窗口保留查看位置。Esc 返回；弹窗支持 Tab 焦点循环。

升级后请重启原来的查看器进程，再刷新浏览器。新版本自动使用 v4 缓存；首次启动会在查看器目录内重建索引，不修改输入项目。旧的后端进程不具备新增接口，不能只刷新网页而不重启服务。

## 数据口径

- 默认使用 `opensfm/reconstruction.topocentric.json`；不存在时使用 `reconstruction.json` 并标为重建文件坐标。相机和点云不混用不同坐标体系。
- 当前 Brown 和 Perspective 相机模型已验证。其他模型仍可显示位置与点云，但视锥是示意形状，误差面板明确标记未支持；不伪装成正确投影。
- `tracks.csv` 是本仓库的二进制 v102 格式，由原生 `pymap` 读取。长度指不同图像的观测数，统计不只包含已经三角化的轨迹。
- 匹配是 `.pkl.gz` 保存的几何验证后结果。空结果不等同于没有共同画面；没有记录不等同于尝试匹配失败。两图的直接匹配与多图 Track 产生的共视关系不同。
- 像素位置用原图尺寸从归一化坐标转换。缩略图不应用 EXIF 自动旋转，保持与特征提取一致。
- 重投影误差由优化后的内参和畸变、`Xcamera = R Xworld + t`、现存三维点计算，单位为原图像素，未按特征 scale 加权。
- 正式误差仅使用 `opensfm/tracks_reconstructed.csv` 中能唯一归属当前分量的观测，覆盖图像统计、点统计与点云着色。`tracks.csv` 用于匹配阶段对照；未保留观测的“最终几何参考残差”不是优化前误差。无最终三维点不计算残差。
- 最终 Tracks 缺失或不可读时显示不可用，不回退匹配 Tracks，不把全部观测判为删除。多分量归属歧义单列，相关点不产生完整质量统计。缺失特征 NPZ 时仍保留 Tracks 自带二维位置。
- 每次观测误差为原图像素欧氏距离；每点平均误差为可计算观测误差的算术平均。顶部中位数/P95 对“每点平均误差”统计，每个点权重相同。阈值默认 3 px，只改变筛选与着色，不删除高误差观测。
- 相机后方、未注册相机、无效投影、缺少二维观测不作为零误差。没有可计算观测的点标记为不可计算、点云中显示灰色。色阶超过 10 px 饱和显示红色，但不截断表格中的真实数值。图像列表中的“观测最大”是该图像各重建点观测误差的最大值，区别于点级均值。
- 未保留不等于误匹配，也可能与注册失败、几何退化、覆盖不足或优化筛选有关。通过变化定位线索，再回看图像。两份 Tracks 必须对应同一次重建。
- 第一版面向已完成的项目。改变项目文件后点重新索引；构建期间源文件变化会使准备失败，避免混用两个版本。缓存以 generation 隔离，旧页面请求不兼容版本时返回明确错误。

## 安装与开发

```bash
bash setup.sh
# 前端源码修改后
cd frontend
npm run build
```

`setup.sh` 使用 `/root/miniconda3/condabin/conda` 创建或使用独立的 `at_viewer` 环境，不向系统 Python 或算法的 `py39` 安装依赖。前端依赖在本目录 `frontend/node_modules`。生产运行不依赖 Vite 开发服务或外部 CDN。

目录：`backend/` 为只读接口与索引器；`frontend/src/` 为 React/Three.js 界面；`frontend/dist/` 为构建产物；`.cache/` 为 SQLite、点云与图像缓存；`validation/` 为测试截图和结果。

首次索引显示进度，后续启动复用缓存。重建索引写入新 generation，完成后原子切换；旧 generation 保留，确认不需要后可在停止服务时人工清理，工具不会自动删除输入数据。

## 验证

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/code/SuperBuild/install/bin/opensfm
export LD_LIBRARY_PATH=/code/SuperBuild/install/lib:/root/miniconda3/envs/at_viewer/lib
/root/miniconda3/envs/at_viewer/bin/python -B -m pytest -q -p no:cacheprovider tests/test_viewer.py tests/test_overlap.py tests/test_comparison.py
/root/miniconda3/envs/at_viewer/bin/python -B tests/real_project_check.py validation/v4/index-3
# 服务启动后；已下载的浏览器放在本目录 .browsers
AT_VIEWER_TEST_URL=http://127.0.0.1:8080 PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" TMPDIR="$PWD/validation/browser-tmp" node tests/browser_smoke.mjs
```

`real_project_check.py` 接受索引目录，独立核对最终观测投影、点级汇总和共视估算。浏览器验证采用无头 Chromium 软件 WebGL，帧率不代表硬件加速性能。


## 诊断接口

- `GET /api/diagnostics?component=0&state=high&threshold=3`：重建点汇总列表。`state` 可为 `high/all/unavailable`，支持 `q` 点 ID 搜索、`sort=mean/median/max/count` 及分页。
- `GET /api/scene/<component>/point/<index>`：三维点坐标、逐点统计及关联观测；索引对应当前 generation 点云顺序。
- `GET /api/scene/<component>/point-errors`：按相同索引排列的 little-endian float64 每点平均误差；NaN 表示不可计算。

验证截图和结果位于 `validation/v2/`，源数据与数值抽查记录位于 `validation/real-project-v2.json`。修改前源码备份位于 `validation/backup-20261016-151034/`。

## 平均重叠率（共视估算）

顶部 Track 数量后显示当前重建分量的平均重叠率，切换分量自动更新。适用于航拍、环绕拍摄和视频抽帧项目；不假设地面为平面，不依赖拍摄顺序。本次不新增实时流接入。

该指标是空间均衡加权的共视比例，不是精确面积重叠率：

1. 仅使用最终 Tracks 中当前分量已注册图像的重建点观测，要求坐标在图内、投影有效、原图重投影误差不超过固定 3 px；页面诊断阈值不影响此指标。不支持投影的相机观测无法参与。
2. 每张图分为 16×16 等面积网格。在每个有有效观测的网格中，计算与另一图共享点数 / 该网格有效点数，再对这些网格等权平均。没有观测的区域不推断为有重叠或无重叠。
3. 两个方向的比例取算术平均作为图像对分数；至少共享 20 个有效三维点才纳入。
4. 每张图取最高的最多 3 个邻居求平均，再对有效图像等权平均。不足 3 个使用实际数量；无有效邻居的图像排除，不记作 0%。全部不可计算时显示“—”。

悬停指标可查看有效图像数/注册图像数和口径。结果受纹理、匹配、重建完整性影响，未观测区域不在统计范围内。视频重复帧可能带来高共视，但高重叠不代表基线充分或三角化质量好。

`GET /api/project` 的 `stats.overlap` 按分量保存 `component`、`metric`、`mean`（0～1 或 null）、`registered_images`、`evaluated_images`、`eligible_pairs` 和 `parameters`。指标版本为 `spatially_balanced_covisibility_final_v2`。v4 缓存会重新索引，结果与 generation 一起切换。升级后重启后端并刷新网页；旧缓存没有此字段时页面显示“—”及升级提示。

独立复算已生成索引（只读源项目）：

```bash
/root/miniconda3/envs/at_viewer/bin/python -B tests/check_overlap_real.py validation/overlap/index
```

天府 v4 验证见 `validation/v4/real-check.json`：166/166 张图像参与，2171 个有效图像对，平均重叠率 36.3%。匹配 Tracks 48,945 条、212,046 次观测；最终 Tracks 33,504 条、132,128 次观测。验证索引独立于服务缓存；重启后端后自动生成 v4 缓存。

## 两阶段比较

- 顶部同时列出匹配与最终 Track 数量；正式重叠率使用最终观测。
- 区分未变、缩短、未保留、关联变化和最终新增；观测未保留分为“点仍在”和“没有最终点”。
- 图像趋势优先按完整 EXIF 拍摄时间排序，否则按输入顺序。显示逐图保留率及五图窗口汇总，不代表迭代优化过程。图像大图支持按特征阶段筛选和逐阶段存活率。
- 图像对在固定节点集合上比较共享 Track 数、连通分量和孤立节点。默认阈值 20；跌破阈值但仍有共视与降至零分别统计。双图对照按 200 条分页。
- API：`/api/comparison/summary`、`/api/comparison/images`、`/api/comparison/tracks`、`/api/comparison/pairs`、`/api/comparison/pairs/<a>/<b>`。支持 `scope=component|project`、`component=0` 及列表筛选分页。
- 缓存版本 4，最终 Tracks 内容指纹参与缓存失效判断；不会修改输入项目。

## 相机匹配关系

选中图像后，三维视图以黄色标记当前分量中与它有有效直接匹配记录的相机。顶部显示全项目、当前分量及分量外的匹配图像数量；侧栏“匹配”按有效匹配数排序，点击“查看匹配”打开双图。分量外匹配单列，不在当前三维视图显示。点或 Track 观测相机仍优先使用紫色。

相机关系不再提供共视模式、共视阈值或两者对照，且不依赖最终 Tracks 文件。Tracks 变化分析、最终重建误差与重叠率统计保留各自口径。重启后端加载 `/api/images/<id>/matching-relations` 接口即可，无需重建 v4 索引。

## 增量式重建过程回放

三维页面点击“复现增量式重建过程”，系统推荐与当前分量对应最完整的记录；手动点击播放开始。支持暂停、前后单步、进度拖动、0.5×/1×/2×/4×速度及定位当前步骤。初始对黄色、当前加入青色、已加入绿色；尚未加入的相机隐藏且不可拾取。点击已出现相机可暂停并查看名称和缩略图。退出恢复原有显示设置。

这是**相机加入顺序回放，使用最终位姿**，不是历史几何快照。默认隐藏点云，可开启“最终点云背景”，它不随步骤变化。多张图像同一事件加入时同时显示，不推测组内次序。

自动发现项目 `log/`、`logs/`、`opensfm/log/`、`opensfm/logs/` 中的 `.log`、`.log.N`、`.txt`，以及 `opensfm` 顶层日志；结构化 `opensfm/reports/reconstruction.json` 可独立作为替代来源。第一版不拼接轮转日志、不解压日志，单文件上限 128 MiB。读取及解析缓存仅在服务内存中，不改动源文件，不要求重建 v4 索引。

多个来源、运行和初始对分支独立列出，不拼接成一次运行。列表标明覆盖、完整性、时间和来源；仅在完整顺序一致时显示日志与报告相互核对一致。失败或未确认的初始对不计入成功加入。缺少初始对的片段不显示完整序号。没有记录、非增量式、不支持的格式及不可读来源分别提示；未映射图像保留文字事件，名称歧义或跨最终分量则不自动进行三维回放。

点击“重新扫描”可发现补入的日志。回放期间每 10 秒重新验证来源；内容变化时暂停并要求重新扫描。切换来源、离开页面、进入浏览器后台会暂停；切换分量或数据 generation 会退出。过程中的相机选择独立于普通匹配与点诊断选择，普通高亮在退出后恢复。

新增只读接口：
- `GET /api/replay/sources?component=0`：来源状态、分支摘要与推荐键。
- `GET /api/replay/events?component=0&source_id=…&run_id=…&branch_id=…&fingerprint=…`：事件序列与映射；来源改变返回 409。只接受系统发现的来源 ID。

验证：
```bash
# 使用前述 at_viewer 环境变量
/root/miniconda3/envs/at_viewer/bin/python -B -m pytest -q -p no:cacheprovider tests/test_replay.py
node tests/replay_state.mjs
/root/miniconda3/envs/at_viewer/bin/python -B tests/check_replay_real.py validation/v4/index-3 --output validation/replay/real-check.json
# 服务重启后，在允许启动 Chromium 的环境运行
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" AT_VIEWER_TEST_URL=http://127.0.0.1:8080 node tests/browser_replay.mjs
```

天府核对：初始对 0099/0100，第 3 张 0101，第 4 张 0097；165 个事件、166 张相机，日志与报告顺序一致。日志结束点数 33,655 与当前模型 33,504 不同，页面明确提示这一差异。真实数据检查记录在 `validation/replay/real-check.json`；当前受限执行环境禁止 Chromium 启动，尚未完成浏览器视觉验收。

## 图像特征筛选与存活率

已移除图像页的网格保留率。特征始终按状态着色，并显示带数量的图例：绿为本图最终保留，红为最终高误差，黄为三维点存在但本图观测未保留，紫为输入 Track 无三维点，蓝为无 Track 记录，灰为状态未确定。选中点保留原颜色，用白色外圈突出。

筛选仅保留四类：全部特征、形成输入 Tracks 的特征、对应已重建三维点、该图最终保留的三维点。第三类包含点存在但本图观测未保留，第四类包含高误差观测。重复关联按特征 ID 去重；同一个特征只要有符合条件的关联即可进入对应筛选。

存活率追踪当前图像、当前重建分量的输入特征链路：全部提取特征 → 输入 Tracks → 输入 Tracks 对应三维点 → 其中本图最终保留。每一步同时显示数量、相对上一阶段的比例及占全部特征的比例。高误差不从最终保留计数中删除。最终新增关联可在三维点筛选中查看，但不混入原输入观测的逐级存活率。无法对应特征文件的输入 ID 排除率计算并单列提示；缺失文件、缺失最终观测、分量归属歧义或零分母显示“—”。

接口 `/api/images/<id>/features?component=…` 新增 `stages` 和 `input_status`，观测新增 `input_track` 来源标记。重启后端并刷新前端即可，无需重建 v4 索引。
