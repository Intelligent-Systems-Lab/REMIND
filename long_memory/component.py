from long_memory.schema import CLASS_SCHEMA
from abc import ABC, abstractmethod
from datetime import datetime
import weaviate
import openai

class Base(ABC):
    # 儲存
    @abstractmethod
    def add_group_memory():
        pass
    @abstractmethod
    def add_single_memory():
        pass
    # 取得
    @abstractmethod
    def get_relavant_memory():
        pass
    # 更新
    @abstractmethod
    def update_group_memory():
        pass
    @abstractmethod
    def update_single_memory():
        pass
    # 刪除
    @abstractmethod
    def del_group_memory():
        pass
    @abstractmethod
    def del_single_memory():
        pass
    @abstractmethod
    def del_all_memory():
        pass

class LongMemory(Base):
    def __init__(self):
        self.children = []
    class Node:
        def __init__(self, text, time=None, vector=None, origin_text=None, embedding_model='openai'):
            self.text = text
            self.origin_text = origin_text
            self.children = []
            if time:
                self.time = time
            else:
                self.time = datetime.now()
            if vector:
                self.vector = vector
            else:
                if embedding_model=='openai':
                    self.embedding_model = openai.embeddings
                else:
                    print('Not support other embedding model now.')
                self.vector = self._create_embedding_vector()
        def _create_embedding_vector(self):
            response = self.embedding_model.create(
                input=self.text,
                model="text-embedding-3-small"
            )
            return response['data'][0]['embedding']
        def add_child(self, child):
            self.children.append(child)
            
    def add_group_memory():
        pass
    def add_single_memory():
        pass
    def get_relavant_memory():
        pass
    def update_group_memory():
        pass
    def update_single_memory():
        pass
    def del_group_memory():
        pass
    def del_single_memory():
        pass
    def del_all_memory():
        pass

class WeaviateLongMemory(Base):
    def __init__(self, weaviate_url, class_name = "base"):
        self.client = weaviate.Client(weaviate_url)
        self.class_name = class_name
        
        if self._class_exists():
            print("Detect existed memory space, loading...")
        else:
            print("Detect empty memory, create memory space...")
            self._create_class(CLASS_SCHEMA)
    # 檢查這個 class 存不存在    
    def _class_exists(self):
        schema = self.client.schema.get()
        classes = [cls['class'] for cls in schema['classes']]
        return self.class_name in classes
    # 創建 class schema
    def _create_class(self, class_schema):
        self.client.schema.create_class(class_schema)
    # 取得本身 shema
    def get_schema(self):
        return self.client.schema.get()
    
    def add_group_memory():
        pass
    def add_single_memory():
        pass
    def get_relavant_memory():
        pass
    def update_group_memory():
        pass
    def update_single_memory():
        pass
    def del_group_memory():
        pass
    def del_single_memory():
        pass
    def del_all_memory():
        pass