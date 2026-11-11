#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <time.h>

int recv_all(int sock, void *buf, int len)
{
    int total = 0, r;
    while (total < len) {
        r = recv(sock, (char *)buf + total, len - total, 0);
        if (r <= 0) return -1;
        total += r;
    }
    return total;
}

int main()
{
    int sock;
    struct sockaddr_in server_addr;

    int n, matrix[20][20];
    char result[100];

    sock = socket(AF_INET, SOCK_STREAM, 0);

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);
    server_addr.sin_addr.s_addr =
        inet_addr("127.0.0.1");

    connect(sock,
            (struct sockaddr *)&server_addr,
            sizeof(server_addr));

    // Enter matrix order
    printf("Enter matrix order: ");
    scanf("%d", &n);

    if (n < 1 || n > 20)
    {
        printf("Invalid matrix order\n");
        close(sock);
        return 1;
    }

    srand(time(NULL));

    // Generate random matrix
    printf("Matrix:\n");

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix[i][j] = rand() % 50 + 1;
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }

    // Send order and matrix
    send(sock, &n, sizeof(n), 0);

    send(sock, matrix, n * n * sizeof(int), 0);

    // Receive classification
    recv_all(sock, result, sizeof(result) - 1);

    printf("Matrix type: %s\n", result);

    close(sock);

    return 0;
}
