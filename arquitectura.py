"""Script para la creacion del diagrama de la arquitectura."""

from diagrams import Diagram, Cluster
from diagrams.aws.analytics import Glue, Quicksight
from diagrams.aws.general import InternetAlt2, User, Users
from diagrams.aws.ml import SagemakerNotebook, SagemakerTrainingJob, SagemakerModel
from diagrams.aws.storage import S3

with Diagram("Arquitectura", show=False):
    users = Users("Usuarios")
    employee = User("MLE")

    source = InternetAlt2("Australia Meteorology site")

    with Cluster("Desarrollo"):
        sm_note = SagemakerNotebook("SM Studio Classic")
        sm_tjob = SagemakerTrainingJob("SM Training Job")
        sm_model = SagemakerModel("XGBoost endpoint")

    with Cluster("Storage"):
        data_s3 = S3("Data")
        glue = Glue("Glue DataBrew")
        results_s3 = S3("Resultados")

    with Cluster("Analytics"):
        qs = Quicksight("Dashboard")

    employee >> sm_note >> sm_tjob >> sm_model
    users >> qs

    source >> glue >> data_s3 >> sm_tjob
    sm_model >> results_s3 >> qs
