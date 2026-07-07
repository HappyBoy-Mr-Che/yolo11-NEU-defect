import sqlite3


class SqliteDBOperator:
    def __init__(self, db_name='./db/user.db'):
        """
        初始化数据库连接，创建游标，并检查和初始化user表结构
        :param db_name: 数据库文件名，默认为'user.db'
        """
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._init_user_table()

    def _init_user_table(self):
        """
        私有方法，用于创建或检查user表结构是否存在，如果不存在则创建
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        );
        """
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def check_user_login(self, username, password):
        """
        查询校验函数，用于检查用户名和密码是否正确
        :param username: 用户名
        :param password: 密码
        :return: 如果用户名和密码匹配返回True，否则返回False
        """
        query = "SELECT * FROM user WHERE username =? AND password =?"
        self.cursor.execute(query, (username, password))
        result = self.cursor.fetchone()
        return result is not None

    def insert_user_data(self, username, password):
        """
        数据入库函数，将新用户的用户名和密码插入到数据库中
        :param username: 用户名
        :param password: 密码
        """
        try:
            insert_query = "INSERT INTO user (username, password) VALUES (?,?)"
            self.cursor.execute(insert_query, (username, password))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"插入数据时出现错误: {e}")

    def close_connection(self):
        """
        关闭数据库连接和游标
        """
        self.cursor.close()
        self.conn.close()
    
    def check_user_exists(self, username):
        """
        校验用户是否已存在
        """
        query = "SELECT COUNT(*) FROM user WHERE username =?"
        self.cursor.execute(query, (username,))
        count = self.cursor.fetchone()[0]
        return count > 0