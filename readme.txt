智能体事故事件重放与反事实遏制实验室

运行测试：
python -m unittest discover -s tests -v

编译检查：
python -m compileall -q replay_lab tests

真实验收：
python -m replay_lab run --fixture fixtures/acceptance.json --workspace out/acceptance

验收完成后可验证报告：
python -m replay_lab verify --workspace out/acceptance

项目仅使用 Python 标准库、SQLite 和本地 JSON，所有夹具均为合成数据。

