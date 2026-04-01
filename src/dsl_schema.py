from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

# 1. РАСШИРЕНИЯ RCPSP
class RCPSPModeRequirement(BaseModel):
    resource_id: str = Field(..., description="ID требуемого ресурса", examples=["R_WORKER"])
    quantity: int = Field(..., description="Требуемое количество ресурса", gt=0)

class RCPSPMode(BaseModel):
    mode_id: str = Field(..., description="Идентификатор режима", examples=["FAST", "CHEAP", "M1"])
    duration: int = Field(..., description="Длительность задачи в данном режиме", ge=0)
    requirements: List[RCPSPModeRequirement] = Field(default_factory=list)
    extensions: Optional[Dict] = Field(None, description="Доп. теги для режима (например, efficiency_note)")

class RCPSPTaskExt(BaseModel):
    location: Optional[str] = Field(None, description="Локация выполнения задачи")
    modes: Optional[List[RCPSPMode]] = Field(None, description="Доступные режимы выполнения")

class RCPSPResourceExt(BaseModel):
    type: Optional[Literal["renewable", "non_renewable"]] = Field(None)
    cost_per_period: Optional[float] = Field(None, description="Стоимость за единицу времени")
    cost_per_unit: Optional[float] = Field(None, description="Стоимость за единицу объема")
    is_zone: Optional[bool] = Field(None, description="Является ли ресурс эксклюзивной зоной")
    is_global_capacity: Optional[bool] = Field(None)
    location_dependent: Optional[bool] = Field(None)
    initial_location: Optional[str] = Field(None)
    transfer_times: Optional[Dict[str, Dict[str, int]]] = Field(None, description="Матрица логистики")

class RCPSPDependencyExt(BaseModel):
    lag: Optional[int] = Field(None, description="Временная задержка связи")

class ObjectiveElement(BaseModel):
    type: str = Field(..., examples=["minimize_makespan", "minimize_cost"])
    weight: float = Field(..., ge=0.0, le=1.0)

class RCPSPProjectExt(BaseModel):
    locations: Optional[List[str]] = Field(None)
    objectives: Optional[List[ObjectiveElement]] = Field(None)


# 2. РАСШИРЕНИЯ CLUSTER
class ClusterRequirement(BaseModel):
    resource_id: str = Field(..., description="ID требуемого типа узла")
    gpu_count: Optional[int] = Field(None)
    cpu_cores: Optional[int] = Field(None)
    ram_gb: Optional[int] = Field(None)

class ClusterTaskExt(BaseModel):
    duration: int = Field(..., description="Оценочное время выполнения в секундах")
    is_preemptible: bool = Field(False, description="Можно ли прерывать джобу")
    requirements: List[ClusterRequirement] = Field(default_factory=list)

class ClusterResourceExt(BaseModel):
    gpu_type: Optional[str] = Field(None, examples=["A100", "V100"])
    gpu_memory_gb: Optional[int] = Field(None)
    ram_gb: Optional[int] = Field(None)
    cpu_cores: Optional[int] = Field(None)


# 3. БАЗОВОЕ ЯДРО
# Модели-контейнеры для расширений
class DependencyExt(BaseModel):
    rcpsp: Optional[RCPSPDependencyExt] = None

class TaskExt(BaseModel):
    rcpsp: Optional[RCPSPTaskExt] = None
    cluster: Optional[ClusterTaskExt] = None

class ResourceExt(BaseModel):
    rcpsp: Optional[RCPSPResourceExt] = None
    cluster: Optional[ClusterResourceExt] = None

class ProjectExt(BaseModel):
    rcpsp: Optional[RCPSPProjectExt] = None


# Основные классы графа
class Dependency(BaseModel):
    task_id: str = Field(..., description="ID предшественника")
    type: Literal["FS", "SS"] = Field(..., description="Тип связи: Finish-to-Start или Start-to-Start")
    extensions: Optional[DependencyExt] = None

class Task(BaseModel):
    id: str = Field(..., description="Уникальный ID задачи")
    name: Optional[str] = None
    dependencies: List[Dependency] = Field(default_factory=list)
    extensions: Optional[TaskExt] = Field(..., description="Доменная специфика задачи")

class Resource(BaseModel):
    id: str = Field(..., description="Уникальный ID ресурса")
    name: Optional[str] = None
    capacity: int = Field(..., description="Базовое количество ресурса")
    extensions: Optional[ResourceExt] = Field(..., description="Доменная специфика ресурса")

class ProjectMeta(BaseModel):
    name: str = Field(..., description="Название проекта")
    objective: str = Field(..., description="Главная цель или 'multi_objective'")

# Корневая модель (Root Model)ы
class SchedulingProblem(BaseModel):
    schema_version: str = Field(default="0.1", description="Версия схемы DSL")
    problem_id: str = Field(..., description="Уникальный ID постановки")
    domain: Literal["rcpsp", "cluster"] = Field(..., description="Предметная область")
    description: Optional[str] = Field(None)
    project: ProjectMeta
    extensions: Optional[ProjectExt] = None
    resources: List[Resource]
    tasks: List[Task]