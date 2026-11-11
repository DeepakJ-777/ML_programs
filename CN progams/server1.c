#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

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
    int sock, client;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len;

    int n, matrix[20][20];
    char result[100];

    sock = socket(AF_INET, SOCK_STREAM, 0);

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(8080);

    bind(sock, (struct sockaddr *)&server_addr,
         sizeof(server_addr));

    listen(sock, 5);

    printf("Server waiting...\n");

    client_len = sizeof(client_addr);

    client = accept(sock,
                    (struct sockaddr *)&client_addr,
                    &client_len);

    // Receive matrix order
    recv_all(client, &n, sizeof(n));

    // Receive matrix
    recv_all(client, matrix, n * n * sizeof(int));

    int upper = 1, lower = 1, diagonal = 1;

    // Check matrix type
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i > j && matrix[i][j] != 0)
                upper = 0;

            if (i < j && matrix[i][j] != 0)
                lower = 0;

            if (i != j && matrix[i][j] != 0)
                diagonal = 0;
        }
    }

    if (diagonal)
        strcpy(result, "Diagonal matrix");
    else if (upper)
        strcpy(result, "Upper triangular matrix");
    else if (lower)
        strcpy(result, "Lower triangular matrix");
    else
        strcpy(result, "None of these");

    // Send result to client
    send(client, result, strlen(result) + 1, 0);

    close(client);
    close(sock);

    return 0;
}
