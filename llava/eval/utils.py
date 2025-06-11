import logging
import time
import os

class Logger:
    def __init__(self):
        # 创建以时间戳命名的日志文件名
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")#
        self.log_filename = f"{self.timestamp}.log"

        # 创建日志目录（可选）
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, self.log_filename)

        # 配置日志系统
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,  # 或 logging.DEBUG / ERROR 等
            format="%(asctime)s - %(levelname)s - %(message)s"#
        )

    def print_log(*args):
        message = ' '.join(str(arg) for arg in args[1:])
        # for i, arg in enumerate(args):
        #     print(f"Argument {i + 1}: {arg}")
        print(message)
        logging.info(message)
        # logging.warning("这是一个警告")
        # logging.error("这是一个错误")