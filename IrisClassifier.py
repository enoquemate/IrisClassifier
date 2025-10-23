import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer  # Correção do import
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class IrisClassifier:
    def __init__(self, custom_csv_path='C:\\Users\\enoqu\\PycharmProjects\\IrisClassifier\\Iris.csv'):
        """
        Inicializa o classificador com o caminho do arquivo CSV personalizado.
        Args:
            custom_csv_path (str): Caminho para o arquivo Iris.csv fornecido.
        """
        self.custom_csv_path = custom_csv_path
        self.df_combined = None
        self.X = None
        self.y = None
        self.model = None
        self.le = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')  # Imputer para tratar NaNs

    def load_data(self):
        """
        Carrega e combina os datasets (Iris.csv e dataset embutido do scikit-learn).
        Remove duplicatas e prepara o DataFrame combinado.
        """
        # Carrega o arquivo personalizado
        df_custom = pd.read_csv(self.custom_csv_path)
        df_custom = df_custom.drop('Id', axis=1)  # Remove a coluna 'Id' se não for necessária

        # Carrega o dataset embutido
        iris = load_iris()
        df_builtin = pd.DataFrame(iris.data, columns=iris.feature_names)
        df_builtin['Species'] = iris.target

        # Combina os datasets
        self.df_combined = pd.concat([df_custom, df_builtin], ignore_index=True)

        # Remove duplicatas com base nas features e espécie
        self.df_combined = self.df_combined.drop_duplicates(
            subset=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'],
            keep='first'
        )
        print(f"Número de amostras após combinação: {len(self.df_combined)}")

    def preprocess_data(self):
        """
        Pré-processa os dados, tratando NaNs, codificando as espécies e separando features e target.
        """
        # Verifica e trata valores ausentes nas features
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.df_combined[features] = self.imputer.fit_transform(self.df_combined[features])

        # Converte 'Species' para numérico, unificando formatos
        self.df_combined['Species'] = self.le.fit_transform(self.df_combined['Species'].astype(str))

        # Define features e target
        self.X = self.df_combined[features]
        self.y = self.df_combined['Species']

    def train_model(self):
        """
        Treina o modelo KNN com os dados preparados.
        """
        # Divide os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Inicializa e treina o modelo
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(X_train, y_train)

        # Faz previsões e calcula acurácia
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Acurácia do modelo: {accuracy:.2f}')

        return y_test, y_pred

    def visualize_results(self, y_test, y_pred):
        """
        Visualiza os resultados com um scatter plot e matriz de confusão.
        Args:
            y_test (array): Valores reais de teste.
            y_pred (array): Valores previstos.
        """
        # Scatter plot de PetalLength vs PetalWidth
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(self.df_combined['PetalLengthCm'], self.df_combined['PetalWidthCm'],
                    c=self.df_combined['Species'], cmap='viridis')
        plt.title('Comprimento vs. Largura de Pétalas por Espécie')
        plt.xlabel('Comprimento Pétala (cm)')
        plt.ylabel('Largura Pétala (cm)')
        plt.colorbar(label='Espécie')

        # Matriz de confusão
        plt.subplot(1, 2, 2)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=self.le.classes_, yticklabels=self.le.classes_)
        plt.title('Matriz de Confusão')
        plt.xlabel('Previsto')
        plt.ylabel('Real')

        plt.tight_layout()
        plt.show()


def main():
    """
    Função principal para executar o projeto.
    """
    # Inicializa o classificador
    classifier = IrisClassifier(
        custom_csv_path='C:\\Users\\enoqu\\PycharmProjects\\IrisClassifier\\Iris.csv')  # Ajuste o caminho
    print("Carregando e combinando dados...")

    # Executa os passos do projeto
    classifier.load_data()
    classifier.preprocess_data()
    y_test, y_pred = classifier.train_model()
    classifier.visualize_results(y_test, y_pred)

    print("Projeto concluído! Verifique os gráficos e a acurácia.")


if __name__ == "__main__":
    main()