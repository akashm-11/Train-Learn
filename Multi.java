 import java.util.*;
 class Main {
    public static void main(String[] args) {
        // Scanner sc = new Scanner(System.in);
        // System.out.println("Enter the matrix1 no.of rows");
        // int n1 = sc.nextInt();
        // System.out.println("Enter the matrix1 no.of cols");
        // int m1 = sc.nextInt();

        // System.out.println("Enter the matrix2 no.of rows");
        // int n2 = sc.nextInt();
        // System.out.println("Enter the matrix2 no.of cols");
        // int m2 = sc.nextInt();
        
        int[] sizes = {3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500};

        for (int size : sizes) {
            float[][] matrix1 = new float[size][size];
            float[][] matrix2 = new float[size][size];
            Random random = new Random();
    
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    matrix1[i][j] = random.nextFloat(2.0f);
                }
            }
    
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    matrix2[i][j] = random.nextFloat(2.0f);
                }
            }
            try {
            long startTime = System.nanoTime();
            float[][] res = multiply(matrix1, matrix2);
            long endTime = System.nanoTime();
            long durationNs = endTime-startTime;
            double durationMs = durationNs / 1_000_000.0;
            System.out.println(durationMs);
            } catch(Exception e) {
                System.out.println("Got an error");
            }
        }
    }

    public static float[][] multiply(float[][] matrix1, float[][] matrix2) {
        int n = matrix1.length;
        int m = matrix2[0].length;

        int n1 = matrix1[0].length;
        int m1 = matrix2.length;

        if (n1 != m1) {
            System.out.println("Matrix multiplication not possible because "+ n1 + " is not equal to "+m1);
            return new float[][]{{}};
        }

        float[][] res = new float[n][m];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                res[i][j] = 0f;

                for (int k = 0; k < m1; k++) {
                    res[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return res;
    }
}