#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>

void print_usage()
{
    printf("Usage: files\n"\
            "\tinfo <file>\n"\
            "\tlink <src> <dst>\n"\
            "\tsymlink <src> <dst>\n"\
            "\trm <file>\n");
}

int PERMISSIONS[] = 
{
    S_IRUSR,
    S_IWUSR,
    S_IXUSR,
    S_IRGRP,
    S_IWGRP,
    S_IXGRP,
    S_IROTH,
    S_IWOTH,
    S_IXOTH
};

int main(int argc, char** argv)
{
    int i, ret;

    // if no args were supplied, print usage
    if (argc == 1)
    {
        print_usage();
        exit(1);
    }

    // if arg info, print file info
    if (strcmp(argv[1], "info") == 0)
    {
        // --get file info--
        // Note: lstat is used instead of stat, so if the file is a symlink
        // it will be stat-ed, not the linked file
        struct stat buf;
        ret = lstat(argv[2], &buf);
        if (ret != -1)
        {
            printf("Inode: %d\n", (int)buf.st_ino); 
            printf("Size: %d\n", (int)buf.st_size); 

            // use PERMISSIONS array to condense logic for printing permissions
            printf("Permissions: ");
            mode_t mode = buf.st_mode;
            for (i = 0; i < sizeof(PERMISSIONS)/sizeof(int); i++)
            {
                if (mode & PERMISSIONS[i])
                {
                    if (i % 3 == 0)
                        printf("r");
                    else if (i % 3 == 1)
                        printf("w");
                    else
                        printf("x");
                }
                else
                {
                    printf("-");
                }    
            } 
            printf("\n\n");
        }
        else
            perror("stat");

        exit(0);
    }
    // if arg link, create a hard link dst to src
    else if (strcmp(argv[1], "link") == 0)
    {
        ret = link(argv[2], argv[3]);
        if (ret == -1)
            perror("link");
        exit(0);
    }
    // if arg symlink, create a soft link dst to src
    else if (strcmp(argv[1], "symlink") == 0)
    {
        ret = symlink(argv[2], argv[3]);
        if (ret == -1)
            perror("symlink");
        exit(0);
    }
    // if arg rm, unlink the given file
    else if (strcmp(argv[1], "rm") == 0)
    {
        ret = unlink(argv[2], argv[3]);
        if (ret == -1)
            perror("unlink");
        exit(0);
    }
    // if no valid args supplied, print usage
    else
    {
        print_usage();
        exit(1);
    }
}
