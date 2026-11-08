/* Minimal winsock smoke test - proves socket programs link and run */
#include <stdio.h>
#include <winsock2.h>

int main(void) {
    WSADATA wsa;
    char hostname[256];

    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        printf("WSAStartup failed: %d\n", WSAGetLastError());
        return 1;
    }

    if (gethostname(hostname, sizeof(hostname)) == 0) {
        printf("Winsock OK. Hostname: %s\n", hostname);
    } else {
        printf("gethostname failed: %d\n", WSAGetLastError());
    }

    WSACleanup();
    return 0;
}
