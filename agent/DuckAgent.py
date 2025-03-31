import os
import duckdb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# 初始化环境
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")
model = "Qwen/QwQ-32B"

client = OpenAI(api_key=openai_api_key,base_url=base_url) if openai_api_key else None

class DuckDBAgent:
    def __init__(self):
        self.conn = None
        self.current_table = None
        self.table_info = None
        self.system_prompt = "你是一个数据分析助手，可以使用工具函数加载CSV数据文件到duckdb数据库中或生成SQL并用工具函数查询数据库。"
    
    def process_natural_language(self, user_input):
        """处理用户自然语言指令"""
        functions = [
            { 'type': 'function',
              'function':{
                "name": "load_data_file",
                "description": "加载CSV文件到DuckDB数据库,并根据文件名作为要创建的表名",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "CSV文件路径，默认从当前目录加载"
                        },
                        "table_name": {
                            "type": "string",
                            "description": "根据加载文件来命名要创建的表名"
                        }
                    },
                    "required": ["file_path","table_name"]
                }
            }},
            {
                'type': 'function',
                
                'function':{
                "name": "generate_sql_query",
                "description": "使用SQL执行查询",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "需要执行SQL查询语句"
                        }
                    },
                    "required": ["sql"]
                }
            }}
        ]
        
        try:
            messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ];
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                top_p=0.95,
                stream=False,
                tools=functions,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            if message.tool_calls:
                function_name = message.tool_calls[0].function.name
                arguments = eval(message.tool_calls[0].function.arguments)
                
                if function_name == "load_data_file":
                    return self._load_data_file(arguments["file_path"], arguments["table_name"])
                elif function_name == "generate_sql_query":
                    return self._execute_sql_query(arguments["sql"])
            else:
                return message.content
                
        except Exception as e:
            return f"处理请求时出错: {str(e)}"
    
    def _load_data_file(self, file_path, table_name):
        """加载数据文件并更新system prompt"""
        try:
            df = pd.read_csv(file_path)
            self.conn = duckdb.connect()
            self.conn.register(table_name, df)
            self.current_table = table_name
            self.table_info = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
            
            # 更新system prompt包含表结构信息
            columns_info = "\n".join([
                f"{row['column_name']} ({row['column_type']})" 
                for _, row in self.table_info.iterrows()
            ])
            self.system_prompt = f"""你是一个数据分析助手，可以加载CSV数据文件或生成SQL并使用工具执行查询。
当前表结构({table_name}):
{columns_info}"""
            
            return f"已成功加载 {file_path} 为表 {table_name}\n表结构:\n{self.table_info.to_string()}"
        except Exception as e:
            return f"加载文件失败: {str(e)}"
    
    def _execute_sql_query(self, sql):
        """执行SQL查询并返回结果"""
        if not self.current_table:
            return "请先加载数据文件"
            
        try:
            result = self.conn.execute(sql).fetchdf()
            return f"查询结果:\n{result.to_string()}"
        except Exception as e:
            return f"执行查询失败: {str(e)}"

# 使用示例
if __name__ == "__main__":
    agent = DuckDBAgent()
    print("欢迎使用自然语言数据查询助手！(输入'exit'退出)")
    
    while True:
        user_input = input("\n用户: ").strip()
        if user_input.lower() == "exit":
            break
            
        response = agent.process_natural_language(user_input)
        print(f"Agent: {response}")
        