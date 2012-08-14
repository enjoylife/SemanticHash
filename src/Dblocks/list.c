#include "list.h"
#include "dbg.h"

List *List_create()
{
	return (List *) calloc(1, sizeof(List));
}

List *List_copy(List *old)
{
	List *new = NULL;
	check_hard(old, "Must pass an initilized list");
	new = List_create();
	check_hard(new, "Failed to allocate memory");
	LIST_FOREACH(old, first, next, cur) {
		List_push(new, cur->value);
	}
	//TODO Speed this up  by copying into a block?
        /*ListNode *new_nodes;
	     ListNode *node;
	     int count = 0;

	     new_nodes = (ListNode *) calloc(old->count, sizeof(ListNode));

	     LIST_FOREACH(old, first, next, cur) {

	     node = new_nodes + (sizeof(ListNode) * count);
	     node->value= cur->value;

	     if(new->last == NULL) {
	         new->first = node;
	         new->last = node;
	     } else {
	         new->last->next = node;
	         node->prev = new->last;
	         new->last = node;
	     }

	     new->count++;
	     count++;
	}*/
	return new;
}

void List_destroy(List *list)
{
	check_hard(list, "Must pass an initilized list");
	LIST_FOREACH(list, first, next, cur) {
		if(cur->prev) {
			free(cur->prev);
		}
	}

	free(list->last);
	free(list);
}

void List_clear(List *list)
{

	check_hard(list, "Must pass an initilized list");
	LIST_FOREACH(list, first, next, cur) {
		free(cur->value);
	}
}


void List_clear_destroy(List *list)
{
	check_hard(list, "Must pass an initilized list");
	LIST_FOREACH(list, first, next, cur) {
		if(cur->prev) {
			free(cur->prev);
			free(cur->value);
		}
	}

	free(list->last);
	free(list);
}

void List_push(List *list, void *value)
{
	check_hard(list, "Must pass an initilized list");
	ListNode *node = calloc(1, sizeof(ListNode));
	check_mem(node);

	node->value = value;

	if(list->last == NULL) {
		list->first = node;
		list->last = node;
	} else {
		list->last->next = node;
		node->prev = list->last;
		list->last = node;
	}

	list->count++;

error:
	return;
}

void *List_pop(List *list)
{
	check_hard(list, "Must pass an initilized list");
	ListNode *node = list->last;
	return node != NULL ? List_remove(list, node) : NULL;
}

void List_shift(List *list, void *value)
{
	check_hard(list, "Must pass an initilized list");
	ListNode *node = calloc(1, sizeof(ListNode));
	check_mem(node);

	node->value = value;

	if(list->first == NULL) {
		list->first = node;
		list->last = node;
	} else {
		node->next = list->first;
		list->first->prev = node;
		list->first = node;
	}

	list->count++;
error:
	return;

}

void *List_unshift(List *list)
{
	check_hard(list, "Must pass an initilized list");
	ListNode *node = list->first;
	return node != NULL ? List_remove(list, node) : NULL;
}

void *List_remove(List *list, ListNode *node)
{
	void *result = NULL;

	check_hard(list, "Must pass an initilized list");
	check(list->first && list->last, "List is empty.");
	check(node, "node can't be NULL");

	if(node == list->first && node == list->last) {
		list->first = NULL;
		list->last = NULL;
	} else if(node == list->first) {
		list->first = node->next;
	} else if (node == list->last) {
		list->last = node->prev;
	} else {
		ListNode *after = node->next;
		ListNode *before = node->prev;
		after->prev = before;
		before->next = after;
	}

	list->count--;
	result = node->value;
	free(node);

error:
	return result;
}

